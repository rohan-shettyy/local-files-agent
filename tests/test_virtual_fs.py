"""Unit tests for Virtual Filesystem models and data structures (ROH-11 Phase 1.1)."""

import pytest
from datetime import datetime
from local_files_agent.virtual_fs import (
    NodeType,
    NodeMetadata,
    TreeNode,
    VirtualTree,
    DepthLimitExceededError,
    InvalidPathError,
    NodeTypeMismatchError,
    NodeNotFoundError,
)


class TestNodeMetadata:
    def test_default_metadata(self):
        meta = NodeMetadata()
        assert isinstance(meta.created_at, datetime)
        assert isinstance(meta.modified_at, datetime)
        assert meta.permissions == "0644"
        assert meta.read_only is False
        assert meta.owner == "agent"
        assert meta.size_bytes == 0

    def test_touch_modified(self):
        meta = NodeMetadata()
        old_modified = meta.modified_at
        meta.touch_modified()
        assert meta.modified_at >= old_modified

    def test_update_size(self):
        meta = NodeMetadata()
        meta.update_size(1024)
        assert meta.size_bytes == 1024
        meta.update_size(-50)
        assert meta.size_bytes == 0


class TestTreeNode:
    def test_create_file_node(self):
        file_node = TreeNode(
            name="notes.txt",
            node_type=NodeType.FILE,
            contents="Hello World!",
        )
        assert file_node.is_file()
        assert not file_node.is_directory()
        assert file_node.contents == "Hello World!"
        assert file_node.metadata.size_bytes == len("Hello World!".encode("utf-8"))

    def test_create_directory_node(self):
        dir_node = TreeNode(
            name="documents",
            node_type=NodeType.DIRECTORY,
        )
        assert dir_node.is_directory()
        assert not dir_node.is_file()
        assert dir_node.metadata.permissions == "0755"
        assert dir_node.contents is None
        assert len(dir_node.children) == 0

    def test_file_cannot_have_children(self):
        with pytest.raises(NodeTypeMismatchError):
            TreeNode(
                name="file.txt",
                node_type=NodeType.FILE,
                children={"sub": TreeNode(name="sub", node_type=NodeType.FILE)},
            )

    def test_directory_cannot_have_contents(self):
        with pytest.raises(NodeTypeMismatchError):
            TreeNode(
                name="folder",
                node_type=NodeType.DIRECTORY,
                contents="Directory content not allowed",
            )

    def test_invalid_node_name(self):
        with pytest.raises(InvalidPathError):
            TreeNode(name="invalid/name.txt", node_type=NodeType.FILE)

    def test_add_and_get_child(self):
        parent = TreeNode(name="root_dir", node_type=NodeType.DIRECTORY)
        child = TreeNode(name="file1.txt", node_type=NodeType.FILE, contents="ABC")
        parent.add_child(child)

        assert parent.get_child("file1.txt") == child
        assert parent.metadata.size_bytes == 3

    def test_remove_child(self):
        parent = TreeNode(name="root_dir", node_type=NodeType.DIRECTORY)
        child = TreeNode(name="file1.txt", node_type=NodeType.FILE, contents="ABC")
        parent.add_child(child)
        assert parent.metadata.size_bytes == 3

        removed = parent.remove_child("file1.txt")
        assert removed == child
        assert parent.get_child("file1.txt") is None
        assert parent.metadata.size_bytes == 0

    def test_get_depth(self):
        root = TreeNode(name="root", node_type=NodeType.DIRECTORY)
        assert root.get_depth() == 0

        l1 = TreeNode(name="l1", node_type=NodeType.DIRECTORY)
        root.add_child(l1)
        assert root.get_depth() == 1

        l2 = TreeNode(name="l2.txt", node_type=NodeType.FILE, contents="Deep")
        l1.add_child(l2)
        assert root.get_depth() == 2

    def test_depth_limit_exceeded_on_add_child(self):
        root = TreeNode(name="root", node_type=NodeType.DIRECTORY)
        curr = root
        for i in range(5):
            node = TreeNode(name=f"dir_{i}", node_type=NodeType.DIRECTORY)
            curr.add_child(node, max_depth=10)
            curr = node

        deep_child = TreeNode(name="too_deep.txt", node_type=NodeType.FILE)
        with pytest.raises(DepthLimitExceededError):
            curr.add_child(deep_child, max_depth=5, current_depth=5)


class TestVirtualTree:
    def test_tree_initialization(self):
        tree = VirtualTree(max_depth=10)
        assert tree.root.name == "/"
        assert tree.root.is_directory()
        assert tree.max_depth == 10
        assert tree.get_total_depth() == 0

    def test_resolve_path(self):
        assert VirtualTree.resolve_path("/") == []
        assert VirtualTree.resolve_path("/a/b/c") == ["a", "b", "c"]
        assert VirtualTree.resolve_path("a/b/../c/./d") == ["a", "c", "d"]

    def test_resolve_path_invalid_chars(self):
        with pytest.raises(InvalidPathError):
            VirtualTree.resolve_path("/invalid:name/file.txt")

    def test_get_node(self):
        tree = VirtualTree()
        folder = TreeNode(name="documents", node_type=NodeType.DIRECTORY)
        doc = TreeNode(name="report.pdf", node_type=NodeType.FILE, contents="PDF Binary Data")
        folder.add_child(doc)
        tree.root.add_child(folder)

        assert tree.get_node("/") == tree.root
        assert tree.get_node("/documents") == folder
        assert tree.get_node("/documents/report.pdf") == doc
        assert tree.get_node("/documents/nonexistent.txt") is None

    def test_get_node_depth(self):
        tree = VirtualTree()
        folder = TreeNode(name="sub1", node_type=NodeType.DIRECTORY)
        file_node = TreeNode(name="sub2.txt", node_type=NodeType.FILE)
        folder.add_child(file_node)
        tree.root.add_child(folder)

        assert tree.get_node_depth("/") == 0
        assert tree.get_node_depth("/sub1") == 1
        assert tree.get_node_depth("/sub1/sub2.txt") == 2

        with pytest.raises(NodeNotFoundError):
            tree.get_node_depth("/nonexistent")

    def test_validate_depth_bound(self):
        tree = VirtualTree(max_depth=2)
        folder1 = TreeNode(name="f1", node_type=NodeType.DIRECTORY)
        folder2 = TreeNode(name="f2", node_type=NodeType.DIRECTORY)
        folder1.add_child(folder2)
        tree.root.add_child(folder1)

        assert tree.validate_depth_bound("/f1/f2") is True
        with pytest.raises(DepthLimitExceededError):
            tree.validate_depth_bound(3)

    def test_to_dict_and_from_dict(self):
        tree = VirtualTree(max_depth=15)
        folder = TreeNode(name="data", node_type=NodeType.DIRECTORY)
        file_node = TreeNode(name="test.txt", node_type=NodeType.FILE, contents="content")
        folder.add_child(file_node)
        tree.root.add_child(folder)

        data = tree.to_dict()
        assert isinstance(data, dict)
        assert data["max_depth"] == 15

        reconstructed = VirtualTree.from_dict(data)
        assert reconstructed.max_depth == 15
        assert reconstructed.get_node("/data/test.txt").contents == "content"
