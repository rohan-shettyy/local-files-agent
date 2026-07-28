"""Virtual tree data structure and node models for in-memory virtual filesystem."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

from local_files_agent.virtual_fs.exceptions import (
    DepthLimitExceededError,
    DirectoryNotEmptyError,
    InvalidPathError,
    NodeAlreadyExistsError,
    NodeNotFoundError,
    NodeTypeMismatchError,
    ReadOnlyError,
)

DEFAULT_MAX_DEPTH = 20


class NodeType(str, Enum):
    """Supported virtual node types."""
    FILE = "file"
    DIRECTORY = "directory"


class NodeMetadata(BaseModel):
    """Metadata attached to a virtual tree node."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    permissions: str = "0644"
    read_only: bool = False
    owner: str = "agent"
    size_bytes: int = 0

    def touch_modified(self) -> None:
        """Update modified_at timestamp to current UTC time."""
        self.modified_at = datetime.now(timezone.utc)

    def update_size(self, size: int) -> None:
        """Update node size in bytes and update modified_at timestamp."""
        self.size_bytes = max(0, size)
        self.touch_modified()


class TreeNode(BaseModel):
    """Virtual tree node representing a directory or a file."""
    name: str
    node_type: NodeType
    contents: Optional[str] = None
    children: Dict[str, "TreeNode"] = Field(default_factory=dict)
    metadata: NodeMetadata = Field(default_factory=NodeMetadata)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate node name format."""
        if not value and value != "":
            raise InvalidPathError("Node name cannot be empty.")
        if "/" in value or "\\" in value:
            if value != "/":
                raise InvalidPathError(f"Node name '{value}' contains invalid path separators.")
        return value

    @model_validator(mode="after")
    def validate_node_type_consistency(self) -> "TreeNode":
        """Enforce node_type invariants for files and directories."""
        if self.node_type == NodeType.FILE:
            if self.children:
                raise NodeTypeMismatchError(f"File node '{self.name}' cannot contain children.")
            if self.metadata.permissions == "0644" and self.metadata.permissions == NodeMetadata().permissions:
                # Default file permissions
                pass
            if self.contents is not None:
                self.metadata.size_bytes = len(self.contents.encode("utf-8"))
            else:
                self.metadata.size_bytes = 0
        elif self.node_type == NodeType.DIRECTORY:
            if self.contents is not None:
                raise NodeTypeMismatchError(f"Directory node '{self.name}' cannot store file contents.")
            if self.metadata.permissions == "0644":
                self.metadata.permissions = "0755"
            self.recalculate_directory_size()
        return self

    def is_file(self) -> bool:
        """Return True if this node is a file."""
        return self.node_type == NodeType.FILE

    def is_directory(self) -> bool:
        """Return True if this node is a directory."""
        return self.node_type == NodeType.DIRECTORY

    def recalculate_directory_size(self) -> int:
        """Recursively calculate total size in bytes for a directory."""
        if self.is_file():
            return self.metadata.size_bytes
        total_size = sum(child.recalculate_directory_size() for child in self.children.values())
        self.metadata.size_bytes = total_size
        return total_size

    def get_child(self, name: str) -> Optional["TreeNode"]:
        """Retrieve child node by name if this node is a directory."""
        if not self.is_directory():
            raise NodeTypeMismatchError(f"Cannot get child from file node '{self.name}'.")
        return self.children.get(name)

    def add_child(
        self,
        child: "TreeNode",
        max_depth: int = DEFAULT_MAX_DEPTH,
        current_depth: int = 0,
    ) -> None:
        """
        Add a child node to this directory while enforcing depth bounds.

        Args:
            child: TreeNode to add.
            max_depth: Maximum allowable depth for the virtual tree.
            current_depth: Depth of this current node from tree root.
        """
        if not self.is_directory():
            raise NodeTypeMismatchError(f"Cannot add child to file node '{self.name}'.")

        child_subtree_depth = child.get_depth()
        new_depth = current_depth + 1 + child_subtree_depth
        if new_depth > max_depth:
            raise DepthLimitExceededError(
                f"Adding node '{child.name}' would result in tree depth {new_depth}, "
                f"exceeding max allowed depth bound of {max_depth}."
            )

        self.children[child.name] = child
        self.recalculate_directory_size()
        self.metadata.touch_modified()

    def remove_child(self, name: str) -> Optional["TreeNode"]:
        """Remove child node by name if this node is a directory."""
        if not self.is_directory():
            raise NodeTypeMismatchError(f"Cannot remove child from file node '{self.name}'.")
        removed = self.children.pop(name, None)
        if removed:
            self.recalculate_directory_size()
            self.metadata.touch_modified()
        return removed

    def get_depth(self) -> int:
        """
        Calculate maximum height/depth of the tree rooted at this node.
        A leaf file or empty directory has depth 0.
        """
        if self.is_file() or not self.children:
            return 0
        return 1 + max(child.get_depth() for child in self.children.values())


# Support recursive Pydantic model resolution
TreeNode.model_rebuild()


class VirtualTree(BaseModel):
    """In-memory virtual filesystem tree data structure."""
    root: TreeNode = Field(
        default_factory=lambda: TreeNode(
            name="/",
            node_type=NodeType.DIRECTORY,
            metadata=NodeMetadata(permissions="0755"),
        )
    )
    max_depth: int = Field(default=DEFAULT_MAX_DEPTH, ge=1, le=100)

    @field_validator("root")
    @classmethod
    def validate_root_is_directory(cls, root_node: TreeNode) -> TreeNode:
        """Ensure root node is a directory."""
        if not root_node.is_directory():
            raise NodeTypeMismatchError("Root node of VirtualTree must be a DIRECTORY.")
        return root_node

    @model_validator(mode="after")
    def validate_root_depth_bound(self) -> "VirtualTree":
        """Verify root tree depth does not exceed max_depth."""
        root_depth = self.root.get_depth()
        if root_depth > self.max_depth:
            raise DepthLimitExceededError(
                f"VirtualTree root depth ({root_depth}) exceeds max depth bound ({self.max_depth})."
            )
        return self

    @staticmethod
    def resolve_path(path: str) -> List[str]:
        """
        Normalize and split a absolute or relative path string into non-empty component names.
        Handles '.' and '..' path segments correctly.
        """
        if not isinstance(path, str):
            raise InvalidPathError(f"Path must be a string, got {type(path)}.")

        clean_path = path.strip()
        if not clean_path or clean_path == "/":
            return []

        parts = [p for p in clean_path.split("/") if p and p != "."]
        resolved: List[str] = []
        for part in parts:
            if part == "..":
                if resolved:
                    resolved.pop()
            else:
                if "\\" in part or ":" in part:
                    raise InvalidPathError(f"Path component '{part}' contains invalid characters.")
                resolved.append(part)
        return resolved

    def get_node(self, path: str) -> Optional[TreeNode]:
        """Navigate to and return node at specified path, or None if not found."""
        parts = self.resolve_path(path)
        current = self.root
        for part in parts:
            if not current.is_directory():
                return None
            child = current.get_child(part)
            if child is None:
                return None
            current = child
        return current

    def get_node_depth(self, path: str) -> int:
        """
        Return depth of node at given path relative to root directory (root is depth 0).
        Raises NodeNotFoundError if path does not exist.
        """
        parts = self.resolve_path(path)
        current = self.root
        depth = 0
        for part in parts:
            if not current.is_directory():
                raise InvalidPathError(f"Path segment '{current.name}' is a file, cannot navigate deeper.")
            child = current.get_child(part)
            if child is None:
                from local_files_agent.virtual_fs.exceptions import NodeNotFoundError
                raise NodeNotFoundError(f"Path '{path}' not found in virtual tree.")
            current = child
            depth += 1
        return depth

    def validate_depth_bound(self, target_path_or_depth: Union[str, int]) -> bool:
        """
        Check whether given path depth or target depth exceeds configured max_depth limit.
        Raises DepthLimitExceededError if limit is breached.
        """
        depth = (
            target_path_or_depth
            if isinstance(target_path_or_depth, int)
            else self.get_node_depth(target_path_or_depth)
        )
        if depth > self.max_depth:
            raise DepthLimitExceededError(
                f"Depth {depth} exceeds max allowed depth bound ({self.max_depth})."
            )
        return True

    def get_total_depth(self) -> int:
        """Return total maximum depth of tree from root."""
        return self.root.get_depth()

    def create(
        self,
        path: str,
        node_type: NodeType = NodeType.FILE,
        contents: Optional[str] = None,
        create_parents: bool = False,
        overwrite: bool = False,
    ) -> TreeNode:
        """
        Create a file or directory at the specified path.

        Args:
            path: Target virtual path.
            node_type: NodeType.FILE or NodeType.DIRECTORY.
            contents: File contents if creating a file.
            create_parents: If True, create missing parent directories.
            overwrite: If True, overwrite existing file at target path.

        Returns:
            The created or updated TreeNode.
        """
        parts = self.resolve_path(path)
        if not parts:
            raise InvalidPathError("Cannot create root directory '/' as it already exists.")

        parent_parts = parts[:-1]
        name = parts[-1]

        current = self.root
        current_path_segments: List[str] = []

        for part in parent_parts:
            current_path_segments.append(part)
            if not current.is_directory():
                raise NodeTypeMismatchError(
                    f"Path segment '/{'/'.join(current_path_segments[:-1])}' is a file, cannot navigate deeper."
                )
            child = current.get_child(part)
            if child is None:
                if not create_parents:
                    raise NodeNotFoundError(
                        f"Parent directory '/{'/'.join(parent_parts)}' does not exist."
                    )
                if current.metadata.read_only:
                    raise ReadOnlyError(
                        f"Cannot create parent directory in read-only directory '/{'/'.join(current_path_segments[:-1])}'."
                    )
                new_dir = TreeNode(name=part, node_type=NodeType.DIRECTORY)
                current.add_child(new_dir, max_depth=self.max_depth, current_depth=len(current_path_segments) - 1)
                current = new_dir
            else:
                if not child.is_directory():
                    raise NodeTypeMismatchError(
                        f"Path segment '/{'/'.join(current_path_segments)}' is a file, not a directory."
                    )
                current = child

        if current.metadata.read_only:
            raise ReadOnlyError(
                f"Cannot create node in read-only directory '/{'/'.join(parent_parts)}'."
            )

        existing = current.get_child(name)
        if existing is not None:
            if not overwrite:
                raise NodeAlreadyExistsError(f"Node already exists at '{path}'.")
            if existing.metadata.read_only:
                raise ReadOnlyError(f"Target node at '{path}' is read-only.")
            if existing.node_type != node_type:
                raise NodeTypeMismatchError(
                    f"Cannot overwrite {existing.node_type.value} with {node_type.value} at '{path}'."
                )
            if node_type == NodeType.FILE:
                existing.contents = contents
                existing.metadata.size_bytes = len((contents or "").encode("utf-8"))
                existing.metadata.touch_modified()
                self.root.recalculate_directory_size()
                return existing
            else:
                existing.metadata.touch_modified()
                return existing

        new_node = TreeNode(name=name, node_type=node_type, contents=contents)
        current.add_child(new_node, max_depth=self.max_depth, current_depth=len(parent_parts))
        self.root.recalculate_directory_size()
        return new_node

    def read(self, path: str) -> Union[str, Dict[str, dict]]:
        """
        Read file contents or list directory info at path.

        Args:
            path: Target virtual path.

        Returns:
            File content string if node is a file, or dictionary of children if node is a directory.
        """
        node = self.get_node(path)
        if node is None:
            raise NodeNotFoundError(f"Path '{path}' not found in virtual tree.")

        if node.is_file():
            return node.contents or ""

        return {
            child_name: {
                "type": child.node_type.value,
                "size_bytes": child.metadata.size_bytes,
                "permissions": child.metadata.permissions,
                "read_only": child.metadata.read_only,
            }
            for child_name, child in node.children.items()
        }

    def read_file(self, path: str) -> str:
        """Read and return string contents of a file at path."""
        node = self.get_node(path)
        if node is None:
            raise NodeNotFoundError(f"Path '{path}' not found in virtual tree.")
        if not node.is_file():
            raise NodeTypeMismatchError(f"Path '{path}' is a directory, not a file.")
        return node.contents or ""

    def list_dir(self, path: str) -> Dict[str, dict]:
        """List children metadata of directory at path."""
        node = self.get_node(path)
        if node is None:
            raise NodeNotFoundError(f"Path '{path}' not found in virtual tree.")
        if not node.is_directory():
            raise NodeTypeMismatchError(f"Path '{path}' is a file, not a directory.")
        return {
            child_name: {
                "type": child.node_type.value,
                "size_bytes": child.metadata.size_bytes,
                "permissions": child.metadata.permissions,
                "read_only": child.metadata.read_only,
            }
            for child_name, child in node.children.items()
        }

    def update(self, path: str, contents: str, mode: str = "overwrite") -> TreeNode:
        """
        Update file contents at path.

        Args:
            path: Target virtual path.
            contents: New content string to write or append.
            mode: "overwrite" or "append".

        Returns:
            Updated TreeNode.
        """
        node = self.get_node(path)
        if node is None:
            raise NodeNotFoundError(f"Path '{path}' not found in virtual tree.")
        if not node.is_file():
            raise NodeTypeMismatchError(f"Cannot update directory '{path}'. Path must refer to a file.")
        if node.metadata.read_only:
            raise ReadOnlyError(f"File '{path}' is read-only.")

        if mode == "overwrite":
            node.contents = contents
        elif mode == "append":
            node.contents = (node.contents or "") + contents
        else:
            raise ValueError(f"Invalid update mode '{mode}'. Expected 'overwrite' or 'append'.")

        node.metadata.size_bytes = len((node.contents or "").encode("utf-8"))
        node.metadata.touch_modified()
        self.root.recalculate_directory_size()
        return node

    def delete(self, path: str, recursive: bool = False) -> TreeNode:
        """
        Delete file or directory node at path.

        Args:
            path: Target virtual path.
            recursive: If True, allow non-empty directory deletion.

        Returns:
            Deleted TreeNode.
        """
        parts = self.resolve_path(path)
        if not parts:
            raise InvalidPathError("Cannot delete root directory '/'.")

        parent_parts = parts[:-1]
        name = parts[-1]

        parent_node = self.get_node("/" + "/".join(parent_parts)) if parent_parts else self.root
        if parent_node is None or not parent_node.is_directory():
            raise NodeNotFoundError(f"Parent directory for '{path}' not found.")

        target_node = parent_node.get_child(name)
        if target_node is None:
            raise NodeNotFoundError(f"Path '{path}' not found in virtual tree.")

        if parent_node.metadata.read_only:
            raise ReadOnlyError(f"Cannot delete node inside read-only directory '/{'/'.join(parent_parts)}'.")

        def _check_read_only(n: TreeNode, current_p: str) -> None:
            if n.metadata.read_only:
                raise ReadOnlyError(f"Cannot delete read-only item '{current_p}'.")
            if n.is_directory():
                for child_n in n.children.values():
                    _check_read_only(child_n, f"{current_p}/{child_n.name}")

        _check_read_only(target_node, path)

        if target_node.is_directory() and target_node.children and not recursive:
            raise DirectoryNotEmptyError(
                f"Directory '{path}' is not empty. Set recursive=True to delete."
            )

        deleted = parent_node.remove_child(name)
        self.root.recalculate_directory_size()
        if deleted is None:
            raise NodeNotFoundError(f"Failed to remove child '{name}' from parent.")
        return deleted

    def move(
        self,
        src_path: str,
        dst_path: str,
        overwrite: bool = False,
    ) -> TreeNode:
        """
        Move or rename file or directory from src_path to dst_path.

        Args:
            src_path: Source path.
            dst_path: Destination path.
            overwrite: If True, overwrite destination file if it exists.

        Returns:
            Moved TreeNode.
        """
        src_parts = self.resolve_path(src_path)
        dst_parts = self.resolve_path(dst_path)

        if not src_parts:
            raise InvalidPathError("Cannot move root directory '/'.")

        src_node = self.get_node(src_path)
        if src_node is None:
            raise NodeNotFoundError(f"Source path '{src_path}' not found.")

        src_parent_parts = src_parts[:-1]
        src_parent = self.get_node("/" + "/".join(src_parent_parts)) if src_parent_parts else self.root
        if src_parent is None or not src_parent.is_directory():
            raise NodeNotFoundError(f"Source parent directory for '{src_path}' not found.")

        if src_parent.metadata.read_only:
            raise ReadOnlyError(f"Cannot move item from read-only directory '/{'/'.join(src_parent_parts)}'.")

        def _check_read_only(n: TreeNode, current_p: str) -> None:
            if n.metadata.read_only:
                raise ReadOnlyError(f"Cannot move read-only item '{current_p}'.")
            if n.is_directory():
                for child_n in n.children.values():
                    _check_read_only(child_n, f"{current_p}/{child_n.name}")

        _check_read_only(src_node, src_path)

        if src_node.is_directory() and len(dst_parts) >= len(src_parts):
            if dst_parts[: len(src_parts)] == src_parts:
                raise InvalidPathError(
                    f"Cannot move directory '{src_path}' into its own subtree '{dst_path}'."
                )

        dst_node = self.get_node(dst_path)

        if dst_node is not None:
            if dst_node.is_directory():
                target_parent = dst_node
                target_name = src_node.name
                target_parent_depth = len(dst_parts)
                target_full_parts = dst_parts + [target_name]

                if target_full_parts == src_parts:
                    return src_node

                existing_child = dst_node.get_child(target_name)
                if existing_child is not None:
                    if not overwrite:
                        raise NodeAlreadyExistsError(
                            f"Target path '/{'/'.join(target_full_parts)}' already exists."
                        )
                    if existing_child.metadata.read_only:
                        raise ReadOnlyError(
                            f"Target path '/{'/'.join(target_full_parts)}' is read-only."
                        )
                    if existing_child.is_directory() and existing_child.children:
                        raise DirectoryNotEmptyError(
                            f"Target directory '/{'/'.join(target_full_parts)}' is not empty."
                        )
                    dst_node.remove_child(target_name)
            else:
                if src_node.is_directory():
                    raise NodeTypeMismatchError(
                        f"Cannot move directory '{src_path}' onto existing file '{dst_path}'."
                    )
                if not overwrite:
                    raise NodeAlreadyExistsError(f"Destination file '{dst_path}' already exists.")
                if dst_node.metadata.read_only:
                    raise ReadOnlyError(f"Destination file '{dst_path}' is read-only.")

                dst_parent_parts = dst_parts[:-1]
                target_parent = (
                    self.get_node("/" + "/".join(dst_parent_parts)) if dst_parent_parts else self.root
                )
                if target_parent is None or not target_parent.is_directory():
                    raise NodeNotFoundError(f"Destination parent directory for '{dst_path}' not found.")
                target_name = dst_parts[-1]
                target_parent_depth = len(dst_parent_parts)

                if dst_parts == src_parts:
                    return src_node

                target_parent.remove_child(target_name)
        else:
            if not dst_parts:
                raise InvalidPathError("Cannot move to root directory '/' as a file/folder name.")

            dst_parent_parts = dst_parts[:-1]
            target_parent = (
                self.get_node("/" + "/".join(dst_parent_parts)) if dst_parent_parts else self.root
            )
            if target_parent is None or not target_parent.is_directory():
                raise NodeNotFoundError(
                    f"Destination parent directory '/{'/'.join(dst_parent_parts)}' does not exist."
                )
            target_name = dst_parts[-1]
            target_parent_depth = len(dst_parent_parts)

        if target_parent.metadata.read_only:
            raise ReadOnlyError(
                f"Cannot move item into read-only directory '/{'/'.join(dst_parent_parts)}'."
            )

        new_depth = target_parent_depth + 1 + src_node.get_depth()
        if new_depth > self.max_depth:
            raise DepthLimitExceededError(
                f"Moving '{src_path}' to target depth {new_depth} exceeds max depth bound of {self.max_depth}."
            )

        src_parent.remove_child(src_node.name)
        src_node.name = target_name
        target_parent.add_child(src_node, max_depth=self.max_depth, current_depth=target_parent_depth)

        self.root.recalculate_directory_size()
        src_parent.metadata.touch_modified()
        target_parent.metadata.touch_modified()

        return src_node

    def to_dict(self) -> dict:
        """Serialize VirtualTree to dictionary representation."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "VirtualTree":
        """Deserialize VirtualTree from dictionary representation."""
        return cls.model_validate(data)

    def to_json_tree(
        self,
        path: str = "/",
        include_metadata: bool = True,
        include_contents: bool = True,
        max_content_length: Optional[int] = None,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Format VirtualTree state into structured JSON observation dictionary."""
        from local_files_agent.virtual_fs.formatter import OutputFormatter
        return OutputFormatter.format_tree_json(
            self,
            path=path,
            include_metadata=include_metadata,
            include_contents=include_contents,
            max_content_length=max_content_length,
            max_depth=max_depth,
        )

    def to_text_tree(
        self,
        path: str = "/",
        max_depth: Optional[int] = None,
        show_metadata: bool = True,
    ) -> str:
        """Format VirtualTree state into ASCII tree diagram string observation."""
        from local_files_agent.virtual_fs.formatter import OutputFormatter
        return OutputFormatter.format_tree_text(
            self,
            path=path,
            max_depth=max_depth,
            show_metadata=show_metadata,
        )

    def format_observation(
        self,
        action: Union[str, Dict[str, Any]],
        success: bool,
        output: Optional[Union[str, Dict[str, Any]]] = None,
        error: Optional[str] = None,
        path: str = "/",
        **kwargs: Any,
    ):
        """Format action execution result alongside VirtualTree observation payload."""
        from local_files_agent.virtual_fs.formatter import OutputFormatter
        return OutputFormatter.format_action_result(
            action=action,
            success=success,
            output=output,
            error=error,
            tree=self,
            path=path,
            **kwargs,
        )

    def clone(self) -> "VirtualTree":
        """
        Create a deep copy of this VirtualTree instance.

        Returns:
            New, isolated VirtualTree instance.
        """
        return VirtualTree.from_dict(self.to_dict())

    def snapshot(
        self,
        snapshot_id: Optional[str] = None,
        label: Optional[str] = None,
    ):
        """
        Create a TreeSnapshot of this VirtualTree instance.

        Returns:
            TreeSnapshot instance.
        """
        from local_files_agent.virtual_fs.snapshot import TreeSnapshot
        return TreeSnapshot.create(self, snapshot_id=snapshot_id, label=label)

    def restore_from_snapshot(self, snapshot: Any) -> "VirtualTree":
        """
        Restore in-place this VirtualTree instance from a TreeSnapshot or dict.

        Args:
            snapshot: TreeSnapshot instance or dictionary representation.

        Returns:
            Self (modified in-place).
        """
        from local_files_agent.virtual_fs.snapshot import TreeSnapshot
        if isinstance(snapshot, dict):
            restored_tree = VirtualTree.from_dict(snapshot)
        elif isinstance(snapshot, TreeSnapshot):
            restored_tree = snapshot.restore()
        elif hasattr(snapshot, "restore"):
            restored_tree = snapshot.restore()
        else:
            raise ValueError(f"Cannot restore from invalid snapshot type: {type(snapshot)}")

        self.root = restored_tree.root
        self.max_depth = restored_tree.max_depth
        return self

    def diff(self, other_tree: "VirtualTree"):
        """
        Compute diff between this tree and another VirtualTree.

        Args:
            other_tree: Target VirtualTree to compare against self.

        Returns:
            TreeDiff instance.
        """
        from local_files_agent.virtual_fs.snapshot import diff_trees
        return diff_trees(self, other_tree)



