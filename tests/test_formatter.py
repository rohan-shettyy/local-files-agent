"""Unit tests for OutputFormatter and ActionResult (ROH-13 Phase 1.3)."""

import pytest
from local_files_agent.virtual_fs import (
    ActionResult,
    NodeType,
    NodeNotFoundError,
    ObservationFormat,
    OutputFormatter,
    VirtualTree,
)


@pytest.fixture
def populated_tree() -> VirtualTree:
    """Fixture providing a populated VirtualTree with files and nested directories."""
    tree = VirtualTree()
    tree.create("/notes.txt", node_type=NodeType.FILE, contents="General notes and ideas.")
    tree.create("/documents/report.pdf", node_type=NodeType.FILE, contents="PDF Binary Data Mock", create_parents=True)
    tree.create("/documents/summary.md", node_type=NodeType.FILE, contents="# Executive Summary\nAll good.", create_parents=True)
    tree.create("/documents/subfolder/empty_dir", node_type=NodeType.DIRECTORY, create_parents=True)
    return tree


class TestOutputFormatterJSON:
    def test_format_tree_json_full(self, populated_tree: VirtualTree):
        json_obs = OutputFormatter.format_tree_json(populated_tree, path="/")
        assert json_obs["name"] == "/"
        assert json_obs["type"] == "directory"
        assert "children" in json_obs
        assert "notes.txt" in json_obs["children"]
        assert "documents" in json_obs["children"]
        assert json_obs["children"]["notes.txt"]["contents"] == "General notes and ideas."
        assert "metadata" in json_obs["children"]["notes.txt"]

    def test_format_tree_json_subtree(self, populated_tree: VirtualTree):
        json_obs = OutputFormatter.format_tree_json(populated_tree, path="/documents")
        assert json_obs["name"] == "documents"
        assert "report.pdf" in json_obs["children"]
        assert "summary.md" in json_obs["children"]

    def test_format_tree_json_exclude_metadata_and_contents(self, populated_tree: VirtualTree):
        json_obs = OutputFormatter.format_tree_json(
            populated_tree, path="/", include_metadata=False, include_contents=False
        )
        assert "metadata" not in json_obs
        assert "contents" not in json_obs["children"]["notes.txt"]

    def test_format_tree_json_content_truncation(self, populated_tree: VirtualTree):
        json_obs = OutputFormatter.format_tree_json(
            populated_tree, path="/notes.txt", max_content_length=10
        )
        assert "truncated" in json_obs["contents"]
        assert json_obs["contents"].startswith("General no")

    def test_format_tree_json_depth_limit(self, populated_tree: VirtualTree):
        json_obs = OutputFormatter.format_tree_json(populated_tree, path="/", max_depth=1)
        sub_docs = json_obs["children"]["documents"]
        assert isinstance(sub_docs["children"], str)
        assert "depth limit" in sub_docs["children"]

    def test_format_tree_json_nonexistent_path(self, populated_tree: VirtualTree):
        with pytest.raises(NodeNotFoundError):
            OutputFormatter.format_tree_json(populated_tree, path="/ghost_folder")


class TestOutputFormatterTreeText:
    def test_format_tree_text_root(self, populated_tree: VirtualTree):
        text_obs = OutputFormatter.format_tree_text(populated_tree, path="/", show_metadata=False)
        assert "/" in text_obs
        assert "documents/" in text_obs
        assert "notes.txt" in text_obs
        assert "report.pdf" in text_obs

    def test_format_tree_text_with_metadata(self, populated_tree: VirtualTree):
        text_obs = OutputFormatter.format_tree_text(populated_tree, path="/", show_metadata=True)
        assert "bytes" in text_obs or "items" in text_obs

    def test_format_tree_text_depth_limit(self, populated_tree: VirtualTree):
        text_obs = OutputFormatter.format_tree_text(populated_tree, path="/", max_depth=1, show_metadata=False)
        assert "documents/" in text_obs
        assert "depth limit" in text_obs

    def test_format_tree_text_nonexistent_path(self, populated_tree: VirtualTree):
        with pytest.raises(NodeNotFoundError):
            OutputFormatter.format_tree_text(populated_tree, path="/invalid")


class TestOutputFormatterFlatList:
    def test_format_flat_list(self, populated_tree: VirtualTree):
        flat_list = OutputFormatter.format_flat_list(populated_tree, path="/")
        paths = [item["path"] for item in flat_list]
        assert "/" in paths
        assert "/notes.txt" in paths
        assert "/documents" in paths
        assert "/documents/report.pdf" in paths
        assert "/documents/subfolder/empty_dir" in paths

    def test_format_flat_list_subtree(self, populated_tree: VirtualTree):
        flat_list = OutputFormatter.format_flat_list(populated_tree, path="/documents")
        paths = [item["path"] for item in flat_list]
        assert "/documents" in paths
        assert "/documents/report.pdf" in paths
        assert "/notes.txt" not in paths


class TestActionResultAndObservation:
    def test_action_result_success(self, populated_tree: VirtualTree):
        res = OutputFormatter.format_action_result(
            action="create('/hello.txt')",
            success=True,
            output="Created /hello.txt",
            tree=populated_tree,
            observation_format=ObservationFormat.JSON,
        )
        assert res.success is True
        assert res.action == "create('/hello.txt')"
        assert res.output == "Created /hello.txt"
        assert res.error is None
        assert isinstance(res.state_observation, dict)

        d = res.to_dict()
        assert d["success"] is True
        assert "timestamp" in d

        prompt_str = res.to_agent_prompt_str()
        assert "[ACTION OUTCOME]" in prompt_str
        assert "Status: SUCCESS" in prompt_str
        assert "[FILESYSTEM STATE OBSERVATION]" in prompt_str

    def test_action_result_failure(self, populated_tree: VirtualTree):
        res = OutputFormatter.format_action_result(
            action="delete('/protected.sys')",
            success=False,
            error="NodeNotFoundError: File not found",
            tree=populated_tree,
            observation_format=ObservationFormat.TREE_TEXT,
            show_metadata=False,
        )
        assert res.success is False
        assert res.error == "NodeNotFoundError: File not found"
        assert isinstance(res.state_observation, str)

        prompt_str = res.to_agent_prompt_str()
        assert "Status: FAILED" in prompt_str
        assert "Error: NodeNotFoundError" in prompt_str

    def test_virtual_tree_convenience_methods(self, populated_tree: VirtualTree):
        json_tree = populated_tree.to_json_tree()
        assert json_tree["name"] == "/"

        text_tree = populated_tree.to_text_tree(show_metadata=False)
        assert "notes.txt" in text_tree

        obs = populated_tree.format_observation(
            action="read('/notes.txt')",
            success=True,
            output="General notes and ideas.",
        )
        assert obs.success is True
        assert obs.output == "General notes and ideas."
