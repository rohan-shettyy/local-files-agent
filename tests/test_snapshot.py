"""Unit tests for Tree State Snapshotting, Reset Engine, and Tree Diffing."""

import json
import pytest
from local_files_agent.virtual_fs import (
    VirtualTree,
    NodeType,
    TreeSnapshot,
    TreeDiff,
    diff_trees,
    count_tree_nodes,
    SnapshotManager,
    ResetEngine,
    SnapshotNotFoundError,
    InvalidSnapshotError,
)


def create_sample_tree() -> VirtualTree:
    """Helper to populate a sample tree structure."""
    tree = VirtualTree()
    tree.create("/Documents/notes.txt", contents="Meeting notes", create_parents=True)
    tree.create("/Documents/report.pdf", contents="Q3 report data", create_parents=True)
    tree.create("/Downloads/installer.dmg", contents="Binary setup", create_parents=True)
    tree.create("/root_file.txt", contents="Root level content")
    return tree


class TestTreeSnapshot:
    def test_snapshot_creation_and_node_counting(self):
        tree = create_sample_tree()
        snap = tree.snapshot(label="test_baseline")

        assert snap.label == "test_baseline"
        assert snap.snapshot_id is not None
        assert snap.file_count == 4
        assert snap.dir_count == 3  # '/', '/Documents', '/Downloads'
        assert snap.node_count == 7
        assert snap.total_size_bytes > 0

    def test_snapshot_restoration_isolation(self):
        tree = create_sample_tree()
        snap = tree.snapshot()

        # Mutate original tree
        tree.create("/Documents/new_file.txt", contents="brand new")
        tree.delete("/root_file.txt")
        tree.update("/Documents/notes.txt", "updated notes content")

        # Restore from snapshot
        restored = snap.restore()

        assert restored.get_node("/root_file.txt") is not None
        assert restored.read_file("/Documents/notes.txt") == "Meeting notes"
        assert restored.get_node("/Documents/new_file.txt") is None

    def test_snapshot_serialization(self):
        tree = create_sample_tree()
        snap = TreeSnapshot.create(tree, snapshot_id="snap-123", label="checkpoint-1")

        # Dict roundtrip
        snap_dict = snap.to_dict()
        snap_from_dict = TreeSnapshot.from_dict(snap_dict)
        assert snap_from_dict.snapshot_id == "snap-123"
        assert snap_from_dict.label == "checkpoint-1"

        # JSON roundtrip
        snap_json = snap.to_json()
        snap_from_json = TreeSnapshot.from_json(snap_json)
        assert snap_from_json.snapshot_id == "snap-123"
        assert snap_from_json.file_count == snap.file_count

        restored_tree = snap_from_json.restore()
        assert restored_tree.read_file("/Documents/notes.txt") == "Meeting notes"

    def test_invalid_snapshot_error(self):
        with pytest.raises(InvalidSnapshotError):
            TreeSnapshot.from_dict({"invalid": "data"})

        with pytest.raises(InvalidSnapshotError):
            TreeSnapshot.from_json("invalid json string {")


class TestVirtualTreeExtensions:
    def test_clone(self):
        tree = create_sample_tree()
        cloned = tree.clone()

        assert cloned is not tree
        assert cloned.read_file("/Documents/notes.txt") == "Meeting notes"

        cloned.delete("/Documents/notes.txt")
        assert tree.get_node("/Documents/notes.txt") is not None
        assert cloned.get_node("/Documents/notes.txt") is None

    def test_restore_from_snapshot_inplace(self):
        tree = create_sample_tree()
        snap = tree.snapshot()

        tree.create("/extra.txt", contents="extra")
        assert tree.get_node("/extra.txt") is not None

        tree.restore_from_snapshot(snap)
        assert tree.get_node("/extra.txt") is None
        assert tree.read_file("/Documents/notes.txt") == "Meeting notes"

    def test_tree_diff_method(self):
        tree1 = create_sample_tree()
        tree2 = tree1.clone()

        diff_identical = tree1.diff(tree2)
        assert diff_identical.is_identical
        assert diff_identical.total_changes == 0

        tree2.create("/new_file.txt", contents="new")
        diff_modified = tree1.diff(tree2)
        assert not diff_modified.is_identical
        assert "/new_file.txt" in diff_modified.added_paths


class TestTreeDiff:
    def test_diff_added_deleted_modified(self):
        base_tree = create_sample_tree()
        target_tree = base_tree.clone()

        # Added
        target_tree.create("/Documents/added.txt", contents="added")
        # Deleted
        target_tree.delete("/Downloads/installer.dmg")
        # Modified
        target_tree.update("/root_file.txt", contents="Modified root content")

        diff = diff_trees(base_tree, target_tree)

        assert not diff.is_identical
        assert diff.added_paths == ["/Documents/added.txt"]
        assert diff.deleted_paths == ["/Downloads/installer.dmg"]
        assert diff.modified_files == ["/root_file.txt"]
        assert diff.total_changes == 3

        summary = diff.summarize()
        assert summary["added_count"] == 1
        assert summary["deleted_count"] == 1
        assert summary["modified_count"] == 1

    def test_diff_moved_files(self):
        base_tree = create_sample_tree()
        target_tree = base_tree.clone()

        # Move file: delete from old path, create at new path with same content
        target_tree.move("/Documents/notes.txt", "/Downloads/notes.txt")

        diff = diff_trees(base_tree, target_tree)

        assert not diff.is_identical
        assert diff.moved_paths == [{"from": "/Documents/notes.txt", "to": "/Downloads/notes.txt"}]
        assert "/Documents/notes.txt" not in diff.deleted_paths
        assert "/Downloads/notes.txt" not in diff.added_paths
        assert diff.total_changes == 1


class TestSnapshotManager:
    def test_manager_crud(self):
        manager = SnapshotManager()
        tree = create_sample_tree()

        snap1 = manager.create_snapshot(tree, snapshot_id="snap1", label="First")
        snap2 = manager.create_snapshot(tree, label="Second")

        assert manager.has_snapshot("snap1")
        assert len(manager.list_snapshots()) == 2

        retrieved = manager.get_snapshot("snap1")
        assert retrieved.label == "First"

        assert manager.delete_snapshot("snap1")
        assert not manager.has_snapshot("snap1")

        manager.clear()
        assert len(manager.list_snapshots()) == 0

    def test_manager_missing_snapshot_error(self):
        manager = SnapshotManager()
        with pytest.raises(SnapshotNotFoundError):
            manager.get_snapshot("non_existent")

    def test_manager_compare_snapshots(self):
        manager = SnapshotManager()
        tree1 = create_sample_tree()
        tree2 = tree1.clone()
        tree2.create("/added.txt", contents="content")

        s1 = manager.create_snapshot(tree1, snapshot_id="base")
        s2 = manager.create_snapshot(tree2, snapshot_id="target")

        diff = manager.compare_snapshots("base", "target")
        assert diff.added_paths == ["/added.txt"]

        diff2 = manager.compare_tree_with_snapshot(tree2, "base")
        assert diff2.added_paths == ["/added.txt"]


class TestResetEngine:
    def test_reset_engine_initialization_and_reset(self):
        base_tree = create_sample_tree()
        engine = ResetEngine(initial_tree=base_tree)

        active = engine.get_active_tree()
        assert active.read_file("/root_file.txt") == "Root level content"

        # Trajectory operations
        active.create("/traj_file.txt", contents="step 1")
        active.update("/root_file.txt", contents="step 2 update")
        engine.increment_step()
        engine.increment_step()

        assert engine.step_count == 2
        diff = engine.get_diff_from_initial()
        assert "/traj_file.txt" in diff.added_paths

        # Reset environment
        reset_tree = engine.reset()

        assert engine.step_count == 0
        assert engine.reset_count == 1
        assert reset_tree.get_node("/traj_file.txt") is None
        assert reset_tree.read_file("/root_file.txt") == "Root level content"

    def test_reset_engine_checkpoints(self):
        base_tree = create_sample_tree()
        engine = ResetEngine(base_tree)

        active = engine.get_active_tree()
        active.create("/ckpt_dir/file.txt", contents="ckpt data", create_parents=True)
        engine.increment_step()

        ckpt = engine.checkpoint(label="after_step_1")

        # Mutate further
        active.delete("/ckpt_dir/file.txt")
        assert active.get_node("/ckpt_dir/file.txt") is None

        # Restore checkpoint
        restored = engine.restore_checkpoint(ckpt.snapshot_id)
        assert restored.read_file("/ckpt_dir/file.txt") == "ckpt data"

    def test_reset_engine_uninitialized_error(self):
        engine = ResetEngine()

        with pytest.raises(InvalidSnapshotError):
            engine.get_active_tree()

        with pytest.raises(InvalidSnapshotError):
            engine.reset()

    def test_high_speed_reset_loop(self):
        base_tree = create_sample_tree()
        engine = ResetEngine(base_tree)

        for _ in range(100):
            active = engine.get_active_tree()
            active.create("/temp_file.txt", contents="temp")
            active.delete("/root_file.txt")
            engine.reset()

        final_tree = engine.get_active_tree()
        assert final_tree.get_node("/temp_file.txt") is None
        assert final_tree.get_node("/root_file.txt") is not None
        assert engine.reset_count == 100
