"""Virtual Filesystem Simulation Module."""

from local_files_agent.virtual_fs.exceptions import (
    VirtualFSError,
    DepthLimitExceededError,
    InvalidPathError,
    NodeTypeMismatchError,
    NodeNotFoundError,
    NodeAlreadyExistsError,
    DirectoryNotEmptyError,
    ReadOnlyError,
    SnapshotError,
    SnapshotNotFoundError,
    InvalidSnapshotError,
)
from local_files_agent.virtual_fs.formatter import (
    ActionResult,
    ObservationFormat,
    OutputFormatter,
)
from local_files_agent.virtual_fs.models import (
    NodeType,
    NodeMetadata,
    TreeNode,
    VirtualTree,
)
from local_files_agent.virtual_fs.snapshot import (
    TreeSnapshot,
    TreeDiff,
    diff_trees,
    count_tree_nodes,
    SnapshotManager,
    ResetEngine,
)

__all__ = [
    "VirtualFSError",
    "DepthLimitExceededError",
    "InvalidPathError",
    "NodeTypeMismatchError",
    "NodeNotFoundError",
    "NodeAlreadyExistsError",
    "DirectoryNotEmptyError",
    "ReadOnlyError",
    "SnapshotError",
    "SnapshotNotFoundError",
    "InvalidSnapshotError",
    "NodeType",
    "NodeMetadata",
    "TreeNode",
    "VirtualTree",
    "ActionResult",
    "ObservationFormat",
    "OutputFormatter",
    "TreeSnapshot",
    "TreeDiff",
    "diff_trees",
    "count_tree_nodes",
    "SnapshotManager",
    "ResetEngine",
]
