"""Unit tests for synthetic dataset validation, sharding, export, and loading pipeline."""

import json
import os
import tempfile
import pytest

from local_files_agent.generator import (
    DatasetExporter,
    DatasetExportError,
    DatasetManifest,
    DatasetValidationError,
    DatasetValidator,
    ValidationResult,
    export_dataset,
    generate_dataset_batch,
    generate_synthetic_prompt,
    generate_unorganized_tree,
    load_sharded_dataset,
    validate_dataset,
)
from local_files_agent.policy.models import PolicyConfig
from local_files_agent.teacher.models import SyntheticPromptSample
from local_files_agent.teacher.providers import TemplateTeacherProvider


@pytest.fixture
def sample_policy() -> PolicyConfig:
    return PolicyConfig(
        allow_delete=False,
        allowed_root="Downloads",
        target_folders=["Receipts", "Screenshots", "Installers"],
        category_rules={
            "Receipts": ["invoice", ".pdf"],
            "Screenshots": ["Screenshot", ".png"],
            "Installers": ["setup", ".dmg"],
        },
        forbidden_paths=["Downloads/.system/config.sys"],
    )


@pytest.fixture
def sample_batch(sample_policy) -> list[SyntheticPromptSample]:
    provider = TemplateTeacherProvider(seed=42)
    return generate_dataset_batch(
        num_samples=10,
        policies=[sample_policy],
        personas=["casual", "terse", "detailed"],
        provider=provider,
        seed=42,
    )


class TestDatasetValidator:
    def test_validate_valid_sample(self, sample_policy):
        tree_output = generate_unorganized_tree(policy=sample_policy, seed=42)
        sample = SyntheticPromptSample(
            prompt="Please clean up my Downloads folder into Receipts and Screenshots.",
            policy=sample_policy,
            persona="casual",
            unorganized_tree_output=tree_output,
        )

        validator = DatasetValidator()
        result = validator.validate_sample(sample)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_invalid_prompt(self, sample_policy):
        tree_output = generate_unorganized_tree(policy=sample_policy, seed=42)
        sample = SyntheticPromptSample(
            prompt="  ",
            policy=sample_policy,
            persona="casual",
            unorganized_tree_output=tree_output,
        )

        validator = DatasetValidator()
        result = validator.validate_sample(sample)

        assert result.is_valid is False
        assert any("prompt is empty" in err for err in result.errors)

    def test_validate_missing_tree_check(self, sample_policy):
        sample = SyntheticPromptSample(
            prompt="Organize files in my Downloads directory.",
            policy=sample_policy,
            persona="terse",
            unorganized_tree_output=None,
        )

        validator = DatasetValidator(check_tree_exists=True)
        result = validator.validate_sample(sample)
        assert result.is_valid is False
        assert any("UnorganizedTreeOutput is missing" in err for err in result.errors)

        validator_lenient = DatasetValidator(check_tree_exists=False)
        result_lenient = validator_lenient.validate_sample(sample)
        assert result_lenient.is_valid is True
        assert any("UnorganizedTreeOutput is missing" in warn for warn in result_lenient.warnings)

    def test_validate_batch_strict_mode(self, sample_policy):
        tree_output = generate_unorganized_tree(policy=sample_policy, seed=42)
        valid_sample = SyntheticPromptSample(
            prompt="Sort files in my Downloads folder.",
            policy=sample_policy,
            persona="casual",
            unorganized_tree_output=tree_output,
        )
        invalid_sample = SyntheticPromptSample(
            prompt="",
            policy=sample_policy,
            persona="casual",
            unorganized_tree_output=tree_output,
        )

        validator = DatasetValidator()
        valid_list, results = validator.validate_batch([valid_sample, invalid_sample], strict=False)

        assert len(valid_list) == 1
        assert len(results) == 2
        assert results[0].is_valid is True
        assert results[1].is_valid is False

        with pytest.raises(DatasetValidationError, match="failed validation"):
            validator.validate_batch([valid_sample, invalid_sample], strict=True)


class TestDatasetExporter:
    def test_export_dataset_jsonl(self, sample_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = DatasetExporter(
                output_dir=tmpdir,
                dataset_name="test_dataset_jsonl",
                format="jsonl",
                max_samples_per_shard=4,
                split_ratios={"train": 0.6, "val": 0.2, "test": 0.2},
                seed=42,
            )

            manifest = exporter.export(sample_batch)

            assert isinstance(manifest, DatasetManifest)
            assert manifest.dataset_name == "test_dataset_jsonl"
            assert manifest.total_samples == 10
            assert manifest.format == "jsonl"
            assert os.path.exists(os.path.join(tmpdir, "manifest.json"))

            # Check splits created
            assert "train" in manifest.splits
            assert "val" in manifest.splits
            assert "test" in manifest.splits

            # Check shard files on disk
            for split_name, split_meta in manifest.splits.items():
                for shard in split_meta.shards:
                    abs_shard_path = os.path.join(tmpdir, shard.file_path)
                    assert os.path.exists(abs_shard_path)
                    assert os.path.getsize(abs_shard_path) == shard.size_bytes

    def test_export_dataset_json_format(self, sample_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = export_dataset(
                samples=sample_batch,
                output_dir=tmpdir,
                dataset_name="test_dataset_json",
                format="json",
                max_samples_per_shard=5,
                split_ratios={"train": 0.8, "val": 0.2},
                seed=123,
            )

            assert manifest.format == "json"
            assert manifest.total_samples == 10

            # Check shard content is valid JSON array
            for split_name, split_meta in manifest.splits.items():
                for shard in split_meta.shards:
                    abs_shard_path = os.path.join(tmpdir, shard.file_path)
                    with open(abs_shard_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        assert isinstance(data, list)
                        assert len(data) == shard.sample_count

    def test_export_empty_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = DatasetExporter(output_dir=tmpdir)
            with pytest.raises(DatasetExportError, match="Cannot export empty dataset"):
                exporter.export([])

    def test_export_invalid_format(self, sample_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unsupported format"):
                DatasetExporter(output_dir=tmpdir, format="parquet")


class TestDatasetLoader:
    def test_load_sharded_dataset_roundtrip(self, sample_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = export_dataset(
                samples=sample_batch,
                output_dir=tmpdir,
                dataset_name="test_roundtrip",
                format="jsonl",
                max_samples_per_shard=3,
                split_ratios={"train": 0.7, "val": 0.3},
                seed=42,
            )

            loaded_train = load_sharded_dataset(tmpdir, split="train")
            assert len(loaded_train) == manifest.splits["train"].sample_count
            assert all(isinstance(s, SyntheticPromptSample) for s in loaded_train)

            loaded_val = load_sharded_dataset(tmpdir, split="val")
            assert len(loaded_val) == manifest.splits["val"].sample_count

            # Test loading all splits
            loaded_all = load_sharded_dataset(tmpdir, split=None)
            assert len(loaded_all) == 10

            # Verify contents match
            train_sample = loaded_train[0]
            assert isinstance(train_sample.prompt, str)
            assert train_sample.policy is not None
            assert train_sample.unorganized_tree_output is not None
            assert train_sample.unorganized_tree_output.tree is not None

    def test_load_checksum_verification_failure(self, sample_batch):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = export_dataset(
                samples=sample_batch,
                output_dir=tmpdir,
                dataset_name="test_checksum",
                format="jsonl",
                max_samples_per_shard=5,
                split_ratios={"train": 1.0},
            )

            # Corrupt shard file
            shard_rel_path = manifest.splits["train"].shards[0].file_path
            shard_abs_path = os.path.join(tmpdir, shard_rel_path)
            with open(shard_abs_path, "a", encoding="utf-8") as f:
                f.write('{"corrupted": true}\n')

            with pytest.raises(DatasetExportError, match="Checksum mismatch"):
                load_sharded_dataset(tmpdir, split="train")


class TestConvenienceFunctions:
    def test_validate_dataset(self, sample_batch):
        valid_samples, results = validate_dataset(sample_batch)
        assert len(valid_samples) == len(sample_batch)
        assert len(results) == len(sample_batch)
        assert all(r.is_valid for r in results)
