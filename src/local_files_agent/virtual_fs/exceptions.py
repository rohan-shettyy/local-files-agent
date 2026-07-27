"""Exceptions for Virtual Filesystem Simulation."""


class VirtualFSError(Exception):
    """Base exception for virtual filesystem errors."""
    pass


class DepthLimitExceededError(VirtualFSError):
    """Raised when tree depth exceeds configured validation bounds."""
    pass


class InvalidPathError(VirtualFSError):
    """Raised when an invalid or malformed path is specified."""
    pass


class NodeTypeMismatchError(VirtualFSError):
    """Raised when operating on a node of wrong type (e.g. adding children to a file)."""
    pass


class NodeNotFoundError(VirtualFSError):
    """Raised when a requested path or node is not found in the virtual tree."""
    pass


class NodeAlreadyExistsError(VirtualFSError):
    """Raised when creating or moving to a target path that already exists."""
    pass


class DirectoryNotEmptyError(VirtualFSError):
    """Raised when attempting non-recursive deletion of a non-empty directory."""
    pass


class ReadOnlyError(VirtualFSError):
    """Raised when attempting to modify a read-only node or directory."""
    pass

