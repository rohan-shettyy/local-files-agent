"""Unit tests for NoiseTreeGenerator and unorganized virtual filesystem tree generation."""

import pytest
from local_files_agent.generator import (
    NoiseTreeConfig,
    NoiseTreeGenerator,
    UnorganizedTreeOutput,
    generate_unorganized_tree,
)
from local_files_agent.policy import PolicyConfig
from local_files_agent.virtual_fs import VirtualTree


def test_default_noise_tree_generation():
    """Test default noise tree generation with default settings."""
    output = generate_unorganized_tree(seed=42)

    assert isinstance(output, UnorganizedTreeOutput)
    assert isinstance(output.tree, VirtualTree)
    assert isinstance(output.policy, PolicyConfig)

    # Verify target folders exist in tree
    for folder in output.policy.target_folders:
        path = f"{output.policy.allowed_root}/{folder}"
        node = output.tree.get_node(path)
        assert node is not None, f"Target directory '{path}' should exist in tree."
        assert node.is_directory()

    # Verify target_files count is within bounds
    assert 5 <= len(output.target_files) <= 15
    assert len(output.noise_files) >= 0


def test_reproducibility_with_seed():
    """Test that setting a seed produces deterministic identical outputs."""
    out1 = generate_unorganized_tree(seed=123)
    out2 = generate_unorganized_tree(seed=123)

    assert len(out1.target_files) == len(out2.target_files)
    assert len(out1.noise_files) == len(out2.noise_files)
    assert len(out1.forbidden_files) == len(out2.forbidden_files)

    for tf1, tf2 in zip(out1.target_files, out2.target_files):
        assert tf1.filename == tf2.filename
        assert tf1.current_path == tf2.current_path
        assert tf1.expected_path == tf2.expected_path


def test_custom_policy_and_config():
    """Test generation using a custom PolicyConfig and NoiseTreeConfig."""
    custom_policy = PolicyConfig(
        allow_delete=False,
        allowed_root="Workspace/Downloads",
        target_folders=["Financial", "Archives"],
        category_rules={
            "Financial": ["invoice", "tax", ".csv"],
            "Archives": [".tar.gz", "backup"],
        },
        forbidden_paths=["Workspace/Downloads/.system/config.sys"],
    )

    custom_config = NoiseTreeConfig(
        allowed_root="Workspace/Downloads",
        min_target_files=3,
        max_target_files=5,
        min_noise_files=2,
        max_noise_files=4,
        min_noise_dirs=1,
        max_noise_dirs=2,
        seed=999,
    )

    generator = NoiseTreeGenerator(config=custom_config)
    output = generator.generate(policy=custom_policy)

    assert 3 <= len(output.target_files) <= 5
    assert 2 <= len(output.noise_files) <= 4

    # Verify custom root target directories
    for folder in custom_policy.target_folders:
        dir_path = f"Workspace/Downloads/{folder}"
        assert output.tree.get_node(dir_path) is not None


def test_mislocated_files_match_policy_rules():
    """Test that target_files strictly match policy category rules and are initially mislocated."""
    output = generate_unorganized_tree(seed=777)
    policy = output.policy

    for tf in output.target_files:
        # Check category resolution
        matched_cat = policy.get_category_for_file(tf.filename)
        assert matched_cat == tf.expected_category, (
            f"File '{tf.filename}' expected category '{tf.expected_category}', but got '{matched_cat}'."
        )

        # Check file exists in tree at current_path
        node = output.tree.get_node(tf.current_path)
        assert node is not None, f"File should exist in tree at '{tf.current_path}'."
        assert node.is_file()

        # Check it is mislocated
        assert tf.current_path != tf.expected_path, (
            f"File '{tf.filename}' should be mislocated, but current_path matches expected_path."
        )


def test_noise_files_do_not_match_categories():
    """Test that generated noise files do not match target policy categories."""
    output = generate_unorganized_tree(seed=555)
    policy = output.policy

    for noise_path in output.noise_files:
        node = output.tree.get_node(noise_path)
        assert node is not None
        filename = node.name
        cat = policy.get_category_for_file(filename)
        assert cat is None, f"Noise file '{filename}' should not match any category, but matched '{cat}'."


def test_forbidden_paths_generation():
    """Test that forbidden system files are populated and marked read-only."""
    output = generate_unorganized_tree(seed=101)

    assert len(output.forbidden_files) > 0
    for forb_path in output.forbidden_files:
        node = output.tree.get_node(forb_path)
        assert node is not None, f"Forbidden file node at '{forb_path}' should exist."
        assert node.metadata.read_only is True


def test_noise_tree_config_validation():
    """Test validation errors for invalid NoiseTreeConfig bounds."""
    with pytest.raises(ValueError, match="min_target_files"):
        NoiseTreeConfig(min_target_files=10, max_target_files=5)

    with pytest.raises(ValueError, match="min_noise_files"):
        NoiseTreeConfig(min_noise_files=8, max_noise_files=2)

    with pytest.raises(ValueError, match="min_noise_dirs"):
        NoiseTreeConfig(min_noise_dirs=5, max_noise_dirs=2)


def test_generate_tree_convenience_method():
    """Test generate_tree helper method returns VirtualTree directly."""
    generator = NoiseTreeGenerator()
    tree = generator.generate_tree(seed=4321)

    assert isinstance(tree, VirtualTree)
    assert tree.get_node("Downloads") is not None
