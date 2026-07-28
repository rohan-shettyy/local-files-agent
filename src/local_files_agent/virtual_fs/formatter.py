"""Response and state output formatter module for agent observations (ROH-13 Phase 1.3)."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from local_files_agent.virtual_fs.exceptions import NodeNotFoundError
from local_files_agent.virtual_fs.models import NodeType, TreeNode, VirtualTree


class ObservationFormat(str, Enum):
    """Supported filesystem observation formats."""
    JSON = "json"
    TREE_TEXT = "tree_text"
    FLAT_LIST = "flat_list"
    COMPACT = "compact"


class ActionResult(BaseModel):
    """Standardized action execution result and state observation payload."""
    success: bool
    action: Union[str, Dict[str, Any]]
    output: Optional[Union[str, Dict[str, Any]]] = None
    error: Optional[str] = None
    state_observation: Optional[Union[Dict[str, Any], str, List[Dict[str, Any]]]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ActionResult to a dictionary."""
        return self.model_dump(mode="json")

    def to_agent_prompt_str(self) -> str:
        """
        Format ActionResult into a clean text prompt block suitable for LLM turn observation context.
        """
        status_str = "SUCCESS" if self.success else "FAILED"
        action_str = self.action if isinstance(self.action, str) else str(self.action)
        
        lines = [
            "[ACTION OUTCOME]",
            f"Status: {status_str}",
            f"Action: {action_str}",
        ]

        if self.output is not None:
            if isinstance(self.output, dict):
                import json
                lines.append(f"Output:\n{json.dumps(self.output, indent=2)}")
            else:
                lines.append(f"Output: {self.output}")

        if self.error:
            lines.append(f"Error: {self.error}")

        if self.state_observation is not None:
            lines.append("\n[FILESYSTEM STATE OBSERVATION]")
            if isinstance(self.state_observation, dict):
                import json
                lines.append(json.dumps(self.state_observation, indent=2))
            elif isinstance(self.state_observation, list):
                import json
                lines.append(json.dumps(self.state_observation, indent=2))
            else:
                lines.append(str(self.state_observation))

        return "\n".join(lines)


class OutputFormatter:
    """
    Formatter returning file content outputs and structured filesystem state representations
    for agent state observations.
    """

    @staticmethod
    def format_node_json(
        node: TreeNode,
        include_metadata: bool = True,
        include_contents: bool = True,
        max_content_length: Optional[int] = None,
        max_depth: Optional[int] = None,
        current_depth: int = 0,
    ) -> Dict[str, Any]:
        """
        Recursively format a TreeNode into a structured JSON dict.

        Args:
            node: Target TreeNode.
            include_metadata: Whether to include metadata (permissions, timestamps, size).
            include_contents: Whether to include file contents.
            max_content_length: Optional length threshold for truncating file contents.
            max_depth: Optional depth limit for recursive traversal.
            current_depth: Internal depth counter.

        Returns:
            Dict representation of the node and its subtree.
        """
        res: Dict[str, Any] = {
            "name": node.name,
            "type": node.node_type.value,
        }

        if include_metadata:
            res["metadata"] = {
                "size_bytes": node.metadata.size_bytes,
                "permissions": node.metadata.permissions,
                "read_only": node.metadata.read_only,
                "owner": node.metadata.owner,
                "created_at": node.metadata.created_at.isoformat(),
                "modified_at": node.metadata.modified_at.isoformat(),
            }

        if node.is_file():
            if include_contents:
                contents = node.contents or ""
                if max_content_length is not None and len(contents) > max_content_length:
                    contents = contents[:max_content_length] + f"... [truncated, total {len(contents)} chars]"
                res["contents"] = contents
        elif node.is_directory():
            if max_depth is not None and current_depth >= max_depth:
                res["children"] = f"... [depth limit {max_depth} reached]"
            else:
                res["children"] = {
                    child_name: OutputFormatter.format_node_json(
                        child,
                        include_metadata=include_metadata,
                        include_contents=include_contents,
                        max_content_length=max_content_length,
                        max_depth=max_depth,
                        current_depth=current_depth + 1,
                    )
                    for child_name, child in sorted(node.children.items())
                }

        return res

    @classmethod
    def format_tree_json(
        cls,
        tree: VirtualTree,
        path: str = "/",
        include_metadata: bool = True,
        include_contents: bool = True,
        max_content_length: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Format VirtualTree or subtree at path into a structured JSON dictionary.

        Args:
            tree: Target VirtualTree.
            path: Root path of observation. Defaults to "/".
            include_metadata: Include node metadata.
            include_contents: Include file contents.
            max_content_length: Truncate long file contents.
            max_depth: Traversal depth limit.

        Returns:
            Dict observation of filesystem state.
        """
        node = tree.get_node(path)
        if node is None:
            raise NodeNotFoundError(f"Path '{path}' not found in virtual tree.")

        return cls.format_node_json(
            node,
            include_metadata=include_metadata,
            include_contents=include_contents,
            max_content_length=max_content_length,
            max_depth=max_depth,
            current_depth=0,
        )

    @classmethod
    def format_tree_text(
        cls,
        tree: VirtualTree,
        path: str = "/",
        max_depth: Optional[int] = None,
        show_metadata: bool = True,
    ) -> str:
        """
        Format VirtualTree or subtree at path into an ASCII tree diagram string.

        Args:
            tree: Target VirtualTree.
            path: Root path of observation. Defaults to "/".
            max_depth: Max depth for tree visualization.
            show_metadata: Include size/type hints in node line.

        Returns:
            ASCII tree diagram string.
        """
        node = tree.get_node(path)
        if node is None:
            raise NodeNotFoundError(f"Path '{path}' not found in virtual tree.")

        lines: List[str] = []

        header_name = node.name if node.name != "/" else "/"
        if node.is_directory() and header_name != "/":
            header_name += "/"

        if show_metadata:
            meta_str = f" ({node.metadata.size_bytes} bytes)" if node.is_file() else f" ({len(node.children)} items, {node.metadata.size_bytes} bytes)"
            lines.append(f"{header_name}{meta_str}")
        else:
            lines.append(header_name)

        if node.is_directory():
            cls._build_text_subtree(
                node=node,
                prefix="",
                lines=lines,
                max_depth=max_depth,
                current_depth=1,
                show_metadata=show_metadata,
            )

        return "\n".join(lines)

    @classmethod
    def _build_text_subtree(
        cls,
        node: TreeNode,
        prefix: str,
        lines: List[str],
        max_depth: Optional[int],
        current_depth: int,
        show_metadata: bool,
    ) -> None:
        """Internal helper for recursive ASCII tree building."""
        if max_depth is not None and current_depth > max_depth:
            lines.append(f"{prefix}└── ... [depth limit {max_depth} reached]")
            return

        children = sorted(node.children.values(), key=lambda n: (n.is_file(), n.name.lower()))
        count = len(children)

        for idx, child in enumerate(children):
            is_last = (idx == count - 1)
            connector = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "

            display_name = child.name + "/" if child.is_directory() else child.name
            if show_metadata:
                if child.is_file():
                    meta_str = f" ({child.metadata.size_bytes} B)"
                else:
                    meta_str = f" ({len(child.children)} items)"
                lines.append(f"{prefix}{connector}{display_name}{meta_str}")
            else:
                lines.append(f"{prefix}{connector}{display_name}")

            if child.is_directory() and child.children:
                cls._build_text_subtree(
                    node=child,
                    prefix=prefix + child_prefix,
                    lines=lines,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                    show_metadata=show_metadata,
                )

    @classmethod
    def format_flat_list(
        cls,
        tree: VirtualTree,
        path: str = "/",
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Format VirtualTree into a flat list of node records containing relative paths and info.

        Args:
            tree: Target VirtualTree.
            path: Root path to list. Defaults to "/".
            include_metadata: Include node metadata details.

        Returns:
            List of node dictionary entries.
        """
        start_node = tree.get_node(path)
        if start_node is None:
            raise NodeNotFoundError(f"Path '{path}' not found in virtual tree.")

        results: List[Dict[str, Any]] = []

        def _traverse(node: TreeNode, current_path: str) -> None:
            item: Dict[str, Any] = {
                "path": current_path,
                "name": node.name,
                "type": node.node_type.value,
            }
            if include_metadata:
                item["size_bytes"] = node.metadata.size_bytes
                item["permissions"] = node.metadata.permissions
                item["read_only"] = node.metadata.read_only
                item["modified_at"] = node.metadata.modified_at.isoformat()
            
            results.append(item)

            if node.is_directory():
                for child_name, child in sorted(node.children.items()):
                    child_path = f"{current_path.rstrip('/')}/{child_name}"
                    _traverse(child, child_path)

        _traverse(start_node, path if path.startswith("/") else "/" + path)
        return results

    @classmethod
    def format_action_result(
        cls,
        action: Union[str, Dict[str, Any]],
        success: bool,
        output: Optional[Union[str, Dict[str, Any]]] = None,
        error: Optional[str] = None,
        tree: Optional[VirtualTree] = None,
        observation_format: ObservationFormat = ObservationFormat.JSON,
        path: str = "/",
        **format_kwargs: Any,
    ) -> ActionResult:
        """
        Construct a standardized ActionResult combining action output with state observation.

        Args:
            action: Description or dict of action performed.
            success: Whether action succeeded.
            output: Result output of action execution (e.g. read contents, list dir).
            error: Error description if success is False.
            tree: VirtualTree instance to capture state observation from.
            observation_format: ObservationFormat format choice.
            path: Root path for state observation.
            **format_kwargs: Extra options passed to formatting method.

        Returns:
            ActionResult instance.
        """
        state_obs: Optional[Union[Dict[str, Any], str, List[Dict[str, Any]]]] = None

        if tree is not None:
            try:
                if observation_format == ObservationFormat.JSON:
                    state_obs = cls.format_tree_json(tree, path=path, **format_kwargs)
                elif observation_format == ObservationFormat.TREE_TEXT:
                    state_obs = cls.format_tree_text(tree, path=path, **format_kwargs)
                elif observation_format == ObservationFormat.FLAT_LIST:
                    state_obs = cls.format_flat_list(tree, path=path, **format_kwargs)
                elif observation_format == ObservationFormat.COMPACT:
                    state_obs = cls.format_tree_json(tree, path=path, include_metadata=False, include_contents=False, **format_kwargs)
            except Exception as e:
                state_obs = f"[State Observation Error: {str(e)}]"

        return ActionResult(
            success=success,
            action=action,
            output=output,
            error=error,
            state_observation=state_obs,
        )
