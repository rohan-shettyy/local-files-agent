"""Unit tests for VirtualTree CRUD Action Handlers (ROH-12 Phase 1.2)."""

import pytest
from local_files_agent.virtual_fs import (
    NodeType,
    VirtualTree,
    DepthLimitExceededError,
    DirectoryNotEmptyError,
    InvalidPathError,
    NodeAlreadyExistsError,
    NodeTypeMismatchError,
    NodeNotFoundError,
    ReadOnlyError,
)


class TestCreateAction:
    def test_create_file_success(self):
        tree = VirtualTree()
        file_node = tree.create("/notes.txt", node_type=NodeType.FILE, contents="Task notes")
        assert file_node.name == "notes.txt"
        assert file_node.contents == "Task notes"
        assert file_node.metadata.size_bytes == len("Task notes".encode("utf-8"))
        assert tree.get_node("/notes.txt") == file_node

    def test_create_directory_success(self):
        tree = VirtualTree()
        dir_node = tree.create("/documents", node_type=NodeType.DIRECTORY)
        assert dir_node.name == "documents"
        assert dir_node.is_directory()
        assert tree.get_node("/documents") == dir_node

    def test_create_nested_file_without_parents_raises_error(self):
        tree = VirtualTree()
        with pytest.raises(NodeNotFoundError):
            tree.create("/a/b/c.txt", node_type=NodeType.FILE, create_parents=False)

    def test_create_nested_file_with_parents(self):
        tree = VirtualTree()
        file_node = tree.create("/a/b/c.txt", node_type=NodeType.FILE, contents="Deep", create_parents=True)
        assert file_node.name == "c.txt"
        assert tree.get_node("/a/b/c.txt") == file_node
        assert tree.get_node("/a/b").is_directory()

    def test_create_duplicate_node_without_overwrite_raises_error(self):
        tree = VirtualTree()
        tree.create("/file.txt", node_type=NodeType.FILE, contents="V1")
        with pytest.raises(NodeAlreadyExistsError):
            tree.create("/file.txt", node_type=NodeType.FILE, contents="V2", overwrite=False)

    def test_create_duplicate_file_with_overwrite(self):
        tree = VirtualTree()
        tree.create("/file.txt", node_type=NodeType.FILE, contents="V1")
        updated = tree.create("/file.txt", node_type=NodeType.FILE, contents="V2 updated", overwrite=True)
        assert updated.contents == "V2 updated"
        assert tree.read_file("/file.txt") == "V2 updated"

    def test_create_root_path_raises_error(self):
        tree = VirtualTree()
        with pytest.raises(InvalidPathError):
            tree.create("/", node_type=NodeType.DIRECTORY)

    def test_create_in_readonly_directory_raises_error(self):
        tree = VirtualTree()
        dir_node = tree.create("/readonly_dir", node_type=NodeType.DIRECTORY)
        dir_node.metadata.read_only = True
        with pytest.raises(ReadOnlyError):
            tree.create("/readonly_dir/new_file.txt", node_type=NodeType.FILE)

    def test_create_exceeds_depth_bound(self):
        tree = VirtualTree(max_depth=2)
        tree.create("/d1/d2", node_type=NodeType.DIRECTORY, create_parents=True)
        with pytest.raises(DepthLimitExceededError):
            tree.create("/d1/d2/d3", node_type=NodeType.DIRECTORY, create_parents=True)


class TestReadAction:
    def test_read_file_content(self):
        tree = VirtualTree()
        tree.create("/info.txt", node_type=NodeType.FILE, contents="System status OK")
        assert tree.read("/info.txt") == "System status OK"
        assert tree.read_file("/info.txt") == "System status OK"

    def test_read_directory_listing(self):
        tree = VirtualTree()
        tree.create("/docs/doc1.txt", node_type=NodeType.FILE, contents="Doc 1", create_parents=True)
        tree.create("/docs/sub", node_type=NodeType.DIRECTORY, create_parents=True)
        listing = tree.read("/docs")
        assert "doc1.txt" in listing
        assert "sub" in listing
        assert listing["doc1.txt"]["type"] == "file"
        assert listing["sub"]["type"] == "directory"

    def test_read_nonexistent_path_raises_error(self):
        tree = VirtualTree()
        with pytest.raises(NodeNotFoundError):
            tree.read("/ghost.txt")

    def test_read_file_on_directory_raises_mismatch(self):
        tree = VirtualTree()
        tree.create("/folder", node_type=NodeType.DIRECTORY)
        with pytest.raises(NodeTypeMismatchError):
            tree.read_file("/folder")


class TestUpdateAction:
    def test_update_file_overwrite(self):
        tree = VirtualTree()
        tree.create("/log.txt", node_type=NodeType.FILE, contents="Line 1\n")
        updated = tree.update("/log.txt", contents="Replaced content", mode="overwrite")
        assert updated.contents == "Replaced content"
        assert tree.read_file("/log.txt") == "Replaced content"

    def test_update_file_append(self):
        tree = VirtualTree()
        tree.create("/log.txt", node_type=NodeType.FILE, contents="Line 1\n")
        updated = tree.update("/log.txt", contents="Line 2\n", mode="append")
        assert updated.contents == "Line 1\nLine 2\n"

    def test_update_directory_raises_error(self):
        tree = VirtualTree()
        tree.create("/data", node_type=NodeType.DIRECTORY)
        with pytest.raises(NodeTypeMismatchError):
            tree.update("/data", contents="some text")

    def test_update_readonly_file_raises_error(self):
        tree = VirtualTree()
        file_node = tree.create("/config.json", node_type=NodeType.FILE, contents="{}")
        file_node.metadata.read_only = True
        with pytest.raises(ReadOnlyError):
            tree.update("/config.json", contents="{'updated': true}")


class TestDeleteAction:
    def test_delete_file_success(self):
        tree = VirtualTree()
        tree.create("/file.txt", node_type=NodeType.FILE, contents="Delete me")
        deleted = tree.delete("/file.txt")
        assert deleted.name == "file.txt"
        assert tree.get_node("/file.txt") is None

    def test_delete_empty_directory_success(self):
        tree = VirtualTree()
        tree.create("/empty_dir", node_type=NodeType.DIRECTORY)
        deleted = tree.delete("/empty_dir")
        assert deleted.name == "empty_dir"
        assert tree.get_node("/empty_dir") is None

    def test_delete_nonempty_directory_without_recursive_raises_error(self):
        tree = VirtualTree()
        tree.create("/dir/file.txt", node_type=NodeType.FILE, create_parents=True)
        with pytest.raises(DirectoryNotEmptyError):
            tree.delete("/dir", recursive=False)

    def test_delete_nonempty_directory_with_recursive_success(self):
        tree = VirtualTree()
        tree.create("/dir/file.txt", node_type=NodeType.FILE, create_parents=True)
        deleted = tree.delete("/dir", recursive=True)
        assert deleted.name == "dir"
        assert tree.get_node("/dir") is None
        assert tree.get_node("/dir/file.txt") is None

    def test_delete_root_raises_error(self):
        tree = VirtualTree()
        with pytest.raises(InvalidPathError):
            tree.delete("/")

    def test_delete_readonly_node_raises_error(self):
        tree = VirtualTree()
        file_node = tree.create("/locked.txt", node_type=NodeType.FILE)
        file_node.metadata.read_only = True
        with pytest.raises(ReadOnlyError):
            tree.delete("/locked.txt")


class TestMoveAction:
    def test_rename_file(self):
        tree = VirtualTree()
        tree.create("/old_name.txt", node_type=NodeType.FILE, contents="Data")
        moved = tree.move("/old_name.txt", "/new_name.txt")
        assert moved.name == "new_name.txt"
        assert tree.get_node("/old_name.txt") is None
        assert tree.read_file("/new_name.txt") == "Data"

    def test_move_file_into_directory(self):
        tree = VirtualTree()
        tree.create("/file.txt", node_type=NodeType.FILE, contents="Data")
        tree.create("/target_dir", node_type=NodeType.DIRECTORY)
        moved = tree.move("/file.txt", "/target_dir")
        assert moved.name == "file.txt"
        assert tree.get_node("/file.txt") is None
        assert tree.read_file("/target_dir/file.txt") == "Data"

    def test_move_directory_into_directory(self):
        tree = VirtualTree()
        tree.create("/src_dir/item.txt", node_type=NodeType.FILE, create_parents=True)
        tree.create("/dst_dir", node_type=NodeType.DIRECTORY)
        moved = tree.move("/src_dir", "/dst_dir")
        assert moved.name == "src_dir"
        assert tree.get_node("/src_dir") is None
        assert tree.read_file("/dst_dir/src_dir/item.txt") == "item.txt" or tree.get_node("/dst_dir/src_dir/item.txt") is not None

    def test_move_directory_inside_itself_raises_error(self):
        tree = VirtualTree()
        tree.create("/dir/sub", node_type=NodeType.DIRECTORY, create_parents=True)
        with pytest.raises(InvalidPathError):
            tree.move("/dir", "/dir/sub/dir")

    def test_move_file_overwrite(self):
        tree = VirtualTree()
        tree.create("/f1.txt", node_type=NodeType.FILE, contents="New Content")
        tree.create("/f2.txt", node_type=NodeType.FILE, contents="Old Content")
        tree.move("/f1.txt", "/f2.txt", overwrite=True)
        assert tree.get_node("/f1.txt") is None
        assert tree.read_file("/f2.txt") == "New Content"

    def test_move_exceeds_max_depth(self):
        tree = VirtualTree(max_depth=3)
        tree.create("/d1/d2", node_type=NodeType.DIRECTORY, create_parents=True)
        tree.create("/deep/file.txt", node_type=NodeType.FILE, create_parents=True)
        # /d1/d2 (depth 2) + /deep/file.txt (subtree height 1) -> target depth 2 + 1 + 1 = 4 > max_depth(3)
        with pytest.raises(DepthLimitExceededError):
            tree.move("/deep", "/d1/d2")
