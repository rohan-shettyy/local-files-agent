"""Tree state snapshotting, reset engine, and tree diffing for Virtual Filesystem Simulation."""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from pydantic import BaseModel, Field

from local_files_agent.virtual_fs.exceptions import (
    InvalidSnapshotError,
    SnapshotNotFoundError,
)
from local_files_agent.virtual_fs.models import TreeNode, VirtualTree


def count_tree_nodes(tree: VirtualTree) -> Dict[str, int]:
    """
    Traverse a VirtualTree and compute node counts and total size.

    Args:
        tree: VirtualTree instance to inspect.

    Returns:
        Dict containing node_count, file_count, dir_count, and total_size_bytes.
    """
    node_count = 0
    file_count = 0
    dir_count = 0
    total_size = 0

    def _walk(node: TreeNode) -> None:
        nonlocal node_count, file_count, dir_count, total_size
        node_count += 1
        if node.is_file():
            file_count += 1
            total_size += node.metadata.size_bytes
        else:
            dir_count += 1
            for child in node.children.values():
                _walk(child)

    _walk(tree.root)
    return {
        "node_count": node_count,
        "file_count": file_count,
        "dir_count": dir_count,
        "total_size_bytes": total_size,
    }


def walk_tree_paths(tree: VirtualTree) -> Dict[str, TreeNode]:
    """
    Recursively collect mapping of path -> TreeNode for all nodes in VirtualTree.

    Args:
        tree: VirtualTree instance.

    Returns:
        Dict mapping absolute virtual path string to corresponding TreeNode.
    """
    paths: Dict[str, TreeNode] = {}

    def _walk(node: TreeNode, current_path: str) -> None:
        paths[current_path] = node
        if node.is_directory():
            for child_name, child_node in node.children.items():
                child_path = "/" + child_name if current_path == "/" else f"{current_path}/{child_name}"
                _walk(child_node, child_path)

    _walk(tree.root, "/")
    return paths


class TreeSnapshot(BaseModel):
    """
    Immutable representation of a VirtualTree snapshot at a specific point in time.
    """
    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    label: Optional[str] = None
    tree_data: Dict[str, Any]
    node_count: int = 0
    file_count: int = 0
    dir_count: int = 0
    total_size_bytes: int = 0

    @classmethod
    def create(
        cls,
        tree: VirtualTree,
        snapshot_id: Optional[str] = None,
        label: Optional[str] = None,
    ) -> "TreeSnapshot":
        """
        Create a TreeSnapshot from a VirtualTree instance.

        Args:
            tree: VirtualTree to snapshot.
            snapshot_id: Optional custom snapshot ID string.
            label: Optional description label for snapshot.

        Returns:
            New TreeSnapshot instance.
        """
        counts = count_tree_nodes(tree)
        serialized_tree = tree.to_dict()
        kwargs: Dict[str, Any] = {
            "tree_data": serialized_tree,
            "node_count": counts["node_count"],
            "file_count": counts["file_count"],
            "dir_count": counts["dir_count"],
            "total_size_bytes": counts["total_size_bytes"],
        }
        if snapshot_id:
            kwargs["snapshot_id"] = snapshot_id
        if label:
            kwargs["label"] = label

        return cls(**kwargs)

    def restore(self) -> VirtualTree:
        """
        Reconstruct and return an isolated VirtualTree instance from stored snapshot.

        Returns:
            New VirtualTree reconstructed from tree_data.
        """
        try:
            return VirtualTree.from_dict(self.tree_data)
        except Exception as err:
            raise InvalidSnapshotError(f"Failed to restore VirtualTree from snapshot '{self.snapshot_id}': {err}") from err

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TreeSnapshot to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeSnapshot":
        """Deserialize TreeSnapshot from dictionary."""
        try:
            return cls.model_validate(data)
        except Exception as err:
            raise InvalidSnapshotError(f"Invalid snapshot data dict: {err}") from err

    def to_json(self) -> str:
        """Serialize TreeSnapshot to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TreeSnapshot":
        """Deserialize TreeSnapshot from JSON string."""
        try:
            return cls.model_validate_json(json_str)
        except Exception as err:
            raise InvalidSnapshotError(f"Invalid snapshot JSON: {err}") from err


class TreeDiff(BaseModel):
    """
    Structured differences between two VirtualTree states or snapshots.
    """
    added_paths: List[str] = Field(default_factory=list)
    deleted_paths: List[str] = Field(default_factory=list)
    modified_files: List[str] = Field(default_factory=list)
    moved_paths: List[Dict[str, str]] = Field(default_factory=list)  # list of {"from": str, "to": str}
    total_changes: int = 0
    is_identical: bool = True

    def summarize(self) -> Dict[str, Any]:
        """Return human-readable summary of tree diff."""
        return {
            "is_identical": self.is_identical,
            "total_changes": self.total_changes,
            "added_count": len(self.added_paths),
            "deleted_count": len(self.deleted_paths),
            "modified_count": len(self.modified_files),
            "moved_count": len(self.moved_paths),
            "added_paths": self.added_paths,
            "deleted_paths": self.deleted_paths,
            "modified_files": self.modified_files,
            "moved_paths": self.moved_paths,
        }


def diff_trees(base_tree: VirtualTree, target_tree: VirtualTree) -> TreeDiff:
    """
    Compute fine-grained structural and content differences between base and target VirtualTrees.

    Args:
        base_tree: The baseline reference VirtualTree.
        target_tree: The modified/current VirtualTree to compare against base.

    Returns:
        TreeDiff containing added, deleted, modified, and moved path records.
    """
    base_map = walk_tree_paths(base_tree)
    target_map = walk_tree_paths(target_tree)

    # Exclude root directory "/" from diff calculations
    base_map.pop("/", None)
    target_map.pop("/", None)

    raw_deleted = set(base_map.keys()) - set(target_map.keys())
    raw_added = set(target_map.keys()) - set(base_map.keys())
    common = set(base_map.keys()) & set(target_map.keys())

    modified_files: List[str] = []
    for path in common:
        b_node = base_map[path]
        t_node = target_map[path]
        if b_node.is_file() and t_node.is_file():
            if b_node.contents != t_node.contents or b_node.metadata.size_bytes != t_node.metadata.size_bytes:
                modified_files.append(path)

    # Detect moved nodes (where content/signature matches between deleted and added)
    moved_paths: List[Dict[str, str]] = []
    matched_deleted: set = set()
    matched_added: set = set()

    for p_del in sorted(raw_deleted):
        b_node = base_map[p_del]
        for p_add in sorted(raw_added):
            if p_add in matched_added:
                continue
            t_node = target_map[p_add]
            if b_node.node_type == t_node.node_type:
                if b_node.is_file():
                    if b_node.contents == t_node.contents:
                        moved_paths.append({"from": p_del, "to": p_add})
                        matched_deleted.add(p_del)
                        matched_added.add(p_add)
                        break
                else:
                    # Directory comparison based on sub-tree structure
                    b_dict = b_node.model_dump(mode="json")
                    t_dict = t_node.model_dump(mode="json")
                    # Ignore name field when checking directory structure equality
                    b_dict.pop("name", None)
                    t_dict.pop("name", None)
                    if b_dict == t_dict:
                        moved_paths.append({"from": p_del, "to": p_add})
                        matched_deleted.add(p_del)
                        matched_added.add(p_add)
                        break

    final_deleted = sorted(list(raw_deleted - matched_deleted))
    final_added = sorted(list(raw_added - matched_added))
    modified_files.sort()

    total_changes = len(final_added) + len(final_deleted) + len(modified_files) + len(moved_paths)
    is_identical = (total_changes == 0)

    return TreeDiff(
        added_paths=final_added,
        deleted_paths=final_deleted,
        modified_files=modified_files,
        moved_paths=moved_paths,
        total_changes=total_changes,
        is_identical=is_identical,
    )


class SnapshotManager:
    """
    Registry for managing, listing, retrieving, and comparing virtual tree snapshots.
    """
    def __init__(self) -> None:
        self._snapshots: Dict[str, TreeSnapshot] = {}

    def create_snapshot(
        self,
        tree: VirtualTree,
        snapshot_id: Optional[str] = None,
        label: Optional[str] = None,
    ) -> TreeSnapshot:
        """
        Take a snapshot of tree and store in registry.

        Args:
            tree: VirtualTree instance to snapshot.
            snapshot_id: Optional snapshot identifier string.
            label: Optional description label.

        Returns:
            Created TreeSnapshot instance.
        """
        snapshot = TreeSnapshot.create(tree, snapshot_id=snapshot_id, label=label)
        self._snapshots[snapshot.snapshot_id] = snapshot
        return snapshot

    def get_snapshot(self, snapshot_id: str) -> TreeSnapshot:
        """
        Retrieve snapshot by ID from registry.

        Raises:
            SnapshotNotFoundError if snapshot_id does not exist.
        """
        if snapshot_id not in self._snapshots:
            raise SnapshotNotFoundError(f"Snapshot with ID '{snapshot_id}' not found in registry.")
        return self._snapshots[snapshot_id]

    def has_snapshot(self, snapshot_id: str) -> bool:
        """Return True if snapshot_id exists in registry."""
        return snapshot_id in self._snapshots

    def list_snapshots(self) -> List[TreeSnapshot]:
        """Return list of all stored snapshots in registry."""
        return list(self._snapshots.values())

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete snapshot by ID from registry.

        Returns:
            True if snapshot was found and deleted, False otherwise.
        """
        if snapshot_id in self._snapshots:
            del self._snapshots[snapshot_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all stored snapshots in registry."""
        self._snapshots.clear()

    def restore_snapshot(self, snapshot_id: str) -> VirtualTree:
        """
        Retrieve and restore VirtualTree from stored snapshot ID.

        Returns:
            New restored VirtualTree instance.
        """
        snapshot = self.get_snapshot(snapshot_id)
        return snapshot.restore()

    def compare_snapshots(self, base_snapshot_id: str, target_snapshot_id: str) -> TreeDiff:
        """
        Compare two stored snapshots by ID and return TreeDiff.
        """
        base_snap = self.get_snapshot(base_snapshot_id)
        target_snap = self.get_snapshot(target_snapshot_id)
        base_tree = base_snap.restore()
        target_tree = target_snap.restore()
        return diff_trees(base_tree, target_tree)

    def compare_tree_with_snapshot(self, tree: VirtualTree, snapshot_id: str) -> TreeDiff:
        """
        Compare active VirtualTree instance against stored baseline snapshot ID.
        """
        snap = self.get_snapshot(snapshot_id)
        base_tree = snap.restore()
        return diff_trees(base_tree, tree)


class ResetEngine:
    """
    State manager and reset engine for VirtualFS environment rollout trajectories.
    """
    def __init__(self, initial_tree: Optional[VirtualTree] = None) -> None:
        self._manager = SnapshotManager()
        self._baseline_snapshot: Optional[TreeSnapshot] = None
        self._active_tree: Optional[VirtualTree] = None
        self._step_count: int = 0
        self._reset_count: int = 0

        if initial_tree is not None:
            self.set_initial_state(initial_tree)

    @property
    def manager(self) -> SnapshotManager:
        """Return internal SnapshotManager instance."""
        return self._manager

    @property
    def step_count(self) -> int:
        """Return current trajectory step count."""
        return self._step_count

    @property
    def reset_count(self) -> int:
        """Return total number of resets performed."""
        return self._reset_count

    def set_initial_state(self, tree: VirtualTree, label: str = "initial_baseline") -> TreeSnapshot:
        """
        Register initial baseline VirtualTree state for environment resets.

        Args:
            tree: VirtualTree instance representing starting environment.
            label: Description label.

        Returns:
            Created baseline TreeSnapshot.
        """
        self._baseline_snapshot = self._manager.create_snapshot(tree, label=label)
        self._active_tree = self._baseline_snapshot.restore()
        self._step_count = 0
        return self._baseline_snapshot

    def get_active_tree(self) -> VirtualTree:
        """
        Return active working VirtualTree instance.

        Raises:
            InvalidSnapshotError if engine has not been initialized with an initial state.
        """
        if self._active_tree is None:
            raise InvalidSnapshotError("ResetEngine active tree is uninitialized. Call set_initial_state() first.")
        return self._active_tree

    def increment_step(self) -> int:
        """Increment active trajectory step counter by 1."""
        self._step_count += 1
        return self._step_count

    def reset(self, snapshot_id: Optional[str] = None) -> VirtualTree:
        """
        Reset active environment tree to baseline initial state or specified snapshot ID.

        Args:
            snapshot_id: Optional snapshot ID to restore from. Defaults to initial baseline snapshot.

        Returns:
            Freshly restored VirtualTree instance.
        """
        if snapshot_id is not None:
            target_snapshot = self._manager.get_snapshot(snapshot_id)
        elif self._baseline_snapshot is not None:
            target_snapshot = self._baseline_snapshot
        else:
            raise InvalidSnapshotError("Cannot reset engine: No baseline snapshot configured.")

        self._active_tree = target_snapshot.restore()
        self._step_count = 0
        self._reset_count += 1
        return self._active_tree

    def checkpoint(self, label: Optional[str] = None) -> TreeSnapshot:
        """
        Take a checkpoint snapshot of current active tree.

        Args:
            label: Optional description label (e.g. 'step_5').

        Returns:
            Created TreeSnapshot instance.
        """
        active_tree = self.get_active_tree()
        ckpt_label = label or f"checkpoint_step_{self._step_count}"
        return self._manager.create_snapshot(active_tree, label=ckpt_label)

    def restore_checkpoint(self, snapshot_id: str) -> VirtualTree:
        """
        Restore active tree state from a specific checkpoint snapshot ID.

        Returns:
            Restored active VirtualTree instance.
        """
        self._active_tree = self._manager.restore_snapshot(snapshot_id)
        return self._active_tree

    def get_diff_from_initial(self) -> TreeDiff:
        """
        Compute tree diff comparing current active tree with initial baseline snapshot.

        Returns:
            TreeDiff detailing changes since baseline reset.
        """
        if self._baseline_snapshot is None:
            raise InvalidSnapshotError("Cannot compute diff: No baseline snapshot configured.")
        active_tree = self.get_active_tree()
        baseline_tree = self._baseline_snapshot.restore()
        return diff_trees(baseline_tree, active_tree)
