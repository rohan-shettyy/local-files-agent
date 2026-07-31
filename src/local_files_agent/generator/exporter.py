"""Dataset validation, serialization, sharding, and export pipeline for RL Fine-Tuning."""

from datetime import datetime, timezone
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field

from local_files_agent.generator.config import TargetFileInfo, UnorganizedTreeOutput
from local_files_agent.generator.exceptions import DatasetExportError, DatasetValidationError
from local_files_agent.policy.models import PolicyConfig
from local_files_agent.teacher.models import SyntheticPromptSample
from local_files_agent.virtual_fs.models import VirtualTree


class ValidationResult(BaseModel):
    """Result of dataset sample validation check."""

    is_valid: bool = Field(description="True if dataset sample passes all validation criteria.")
    errors: List[str] = Field(default_factory=list, description="List of fatal validation error messages.")
    warnings: List[str] = Field(default_factory=list, description="List of non-fatal warning messages.")
    sample_id: Optional[str] = Field(default=None, description="Optional sample identifier.")


class DatasetValidator:
    """Validator for verifying prompt-policy-tree triplets before dataset export."""

    def __init__(self, min_prompt_len: int = 5, check_tree_exists: bool = True):
        self.min_prompt_len = min_prompt_len
        self.check_tree_exists = check_tree_exists

    def validate_sample(
        self, sample: SyntheticPromptSample, sample_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a single SyntheticPromptSample item.

        Args:
            sample: SyntheticPromptSample item.
            sample_id: Optional identifier string.

        Returns:
            ValidationResult instance detailing pass/fail status and messages.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Prompt check
        if not isinstance(sample.prompt, str) or not sample.prompt.strip():
            errors.append("Sample prompt is empty or not a string.")
        elif len(sample.prompt.strip()) < self.min_prompt_len:
            errors.append(
                f"Sample prompt length ({len(sample.prompt.strip())}) is less than min threshold ({self.min_prompt_len})."
            )

        # 2. Policy check
        if sample.policy is None:
            errors.append("Sample policy is missing.")
        elif not isinstance(sample.policy, PolicyConfig):
            errors.append(f"Sample policy is invalid type: {type(sample.policy)}.")
        else:
            if not sample.policy.allowed_root or not sample.policy.allowed_root.strip():
                errors.append("Policy allowed_root is empty.")
            if not sample.policy.target_folders:
                warnings.append("Policy has empty target_folders.")

        # 3. Unorganized tree check
        if sample.unorganized_tree_output is None:
            if self.check_tree_exists:
                errors.append("UnorganizedTreeOutput is missing from sample.")
            else:
                warnings.append("UnorganizedTreeOutput is missing from sample.")
        else:
            tree_output = sample.unorganized_tree_output
            if not isinstance(tree_output.tree, VirtualTree):
                errors.append("UnorganizedTreeOutput tree is not a valid VirtualTree instance.")
            else:
                # Check root node exists in virtual tree
                root_parts = VirtualTree.resolve_path(sample.policy.allowed_root)
                if root_parts:
                    node = tree_output.tree.get_node(sample.policy.allowed_root)
                    if node is None:
                        warnings.append(
                            f"Allowed root path '{sample.policy.allowed_root}' was not found in initial tree."
                        )

            # Check target files consistency
            if not tree_output.target_files:
                warnings.append("No target files specified in unorganized_tree_output.")
            else:
                for tf in tree_output.target_files:
                    if not tf.filename or not tf.current_path:
                        errors.append(f"Target file entry missing filename or current_path: {tf}")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            sample_id=sample_id,
        )

    def validate_batch(
        self,
        samples: List[SyntheticPromptSample],
        strict: bool = False,
    ) -> Tuple[List[SyntheticPromptSample], List[ValidationResult]]:
        """
        Validate a list of synthetic dataset samples.

        Args:
            samples: List of SyntheticPromptSample items.
            strict: If True, raises DatasetValidationError upon encountering any invalid sample.

        Returns:
            Tuple of (valid_samples, all_validation_results).
        """
        valid_samples: List[SyntheticPromptSample] = []
        results: List[ValidationResult] = []

        for idx, sample in enumerate(samples):
            sid = sample.metadata.get("id") or f"sample_{idx:05d}"
            res = self.validate_sample(sample, sample_id=sid)
            results.append(res)

            if res.is_valid:
                valid_samples.append(sample)
            elif strict:
                raise DatasetValidationError(
                    f"Dataset sample [{sid}] failed validation: {'; '.join(res.errors)}"
                )

        return valid_samples, results


class ShardMetadata(BaseModel):
    """Metadata detailing a single exported dataset shard file."""

    file_path: str = Field(description="Relative file path to shard from output directory.")
    sample_count: int = Field(description="Number of dataset samples contained in this shard.")
    size_bytes: int = Field(description="Size of shard file in bytes.")
    checksum_sha256: str = Field(description="SHA256 checksum hex digest of the shard file.")


class SplitMetadata(BaseModel):
    """Metadata detailing an exported dataset split (e.g. train, val, test)."""

    sample_count: int = Field(description="Total number of samples in this split.")
    shard_count: int = Field(description="Number of shard files created for this split.")
    shards: List[ShardMetadata] = Field(default_factory=list, description="List of shard metadata objects.")


class DatasetManifest(BaseModel):
    """Dataset manifest containing global metadata, split information, and checksums."""

    dataset_name: str = Field(description="Name/identifier of the exported dataset.")
    version: str = Field(default="1.0", description="Dataset schema version identifier.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp.",
    )
    total_samples: int = Field(description="Total valid dataset samples exported across all splits.")
    invalid_samples_count: int = Field(description="Count of samples skipped due to validation failure.")
    format: str = Field(description="File format used for shards ('jsonl' or 'json').")
    split_ratios: Dict[str, float] = Field(description="Target split ratios configured for export.")
    splits: Dict[str, SplitMetadata] = Field(default_factory=dict, description="Dictionary mapping split names to SplitMetadata.")
    seed: Optional[int] = Field(default=None, description="Random seed used for deterministic shuffling/splitting.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional custom dataset metadata.")

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps(self.model_dump(mode="json"), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetManifest":
        """Deserialize manifest from JSON string."""
        data = json.loads(json_str)
        return cls.model_validate(data)


class DatasetExporter:
    """
    Validates, serializes, splits, and shards synthetic prompt-policy-tree dataset items into standardized format.
    """

    def __init__(
        self,
        output_dir: str,
        dataset_name: str = "filesystem_rl_synthetic_dataset",
        format: str = "jsonl",
        max_samples_per_shard: int = 100,
        split_ratios: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 42,
        validate: bool = True,
        strict_validation: bool = False,
    ):
        """
        Initialize DatasetExporter.

        Args:
            output_dir: Directory where sharded dataset and manifest.json will be saved.
            dataset_name: Identifier name for the dataset.
            format: Output serialization file format ("jsonl" or "json").
            max_samples_per_shard: Maximum samples per shard file.
            split_ratios: Dict mapping split names ("train", "val", "test") to ratios. Defaults to {"train": 0.8, "val": 0.1, "test": 0.1}.
            seed: Integer seed for reproducible shuffling and splitting.
            validate: Whether to run validation prior to export.
            strict_validation: If True, raise exception on any invalid sample.
        """
        if format not in ("jsonl", "json"):
            raise ValueError(f"Unsupported format '{format}'. Supported formats: 'jsonl', 'json'.")

        if max_samples_per_shard < 1:
            raise ValueError(f"max_samples_per_shard must be >= 1, got {max_samples_per_shard}.")

        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.format = format.lower()
        self.max_samples_per_shard = max_samples_per_shard
        self.split_ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
        self.seed = seed
        self.validate = validate
        self.strict_validation = strict_validation
        self.validator = DatasetValidator()

    def export(
        self,
        samples: List[SyntheticPromptSample],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetManifest:
        """
        Export a list of SyntheticPromptSample items into sharded dataset files and write manifest.json.

        Args:
            samples: List of SyntheticPromptSample items.
            extra_metadata: Optional dict of extra metadata to attach to the manifest.

        Returns:
            DatasetManifest object describing the exported dataset.
        """
        if not samples:
            raise DatasetExportError("Cannot export empty dataset (samples list is empty).")

        # 1. Validation step
        invalid_count = 0
        if self.validate:
            valid_samples, validation_results = self.validator.validate_batch(
                samples, strict=self.strict_validation
            )
            invalid_count = len(samples) - len(valid_samples)
        else:
            valid_samples = list(samples)

        if not valid_samples:
            raise DatasetExportError(
                f"All {len(samples)} dataset samples failed validation. Cannot export dataset."
            )

        # 2. Shuffle deterministically
        shuffled = list(valid_samples)
        if self.seed is not None:
            rng = random.Random(self.seed)
            rng.shuffle(shuffled)

        # 3. Partition samples by split
        splits_samples = self._partition_splits(shuffled, self.split_ratios)

        # 4. Write shards and record metadata
        os.makedirs(self.output_dir, exist_ok=True)
        split_metadata_dict: Dict[str, SplitMetadata] = {}

        for split_name, split_samples in splits_samples.items():
            if not split_samples:
                continue

            split_dir = os.path.join(self.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            shard_metadatas: List[ShardMetadata] = []
            num_shards = (len(split_samples) + self.max_samples_per_shard - 1) // self.max_samples_per_shard

            for shard_idx in range(num_shards):
                start = shard_idx * self.max_samples_per_shard
                end = min(start + self.max_samples_per_shard, len(split_samples))
                shard_items = split_samples[start:end]

                file_name = f"shard_{shard_idx:05d}.{self.format}"
                abs_file_path = os.path.join(split_dir, file_name)
                rel_file_path = os.path.join(split_name, file_name)

                # Write shard file
                size_bytes, sha256_hex = self._write_shard_file(abs_file_path, shard_items)

                shard_metadatas.append(
                    ShardMetadata(
                        file_path=rel_file_path,
                        sample_count=len(shard_items),
                        size_bytes=size_bytes,
                        checksum_sha256=sha256_hex,
                    )
                )

            split_metadata_dict[split_name] = SplitMetadata(
                sample_count=len(split_samples),
                shard_count=len(shard_metadatas),
                shards=shard_metadatas,
            )

        # 5. Build and save manifest
        manifest = DatasetManifest(
            dataset_name=self.dataset_name,
            version="1.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            total_samples=len(valid_samples),
            invalid_samples_count=invalid_count,
            format=self.format,
            split_ratios=self.split_ratios,
            splits=split_metadata_dict,
            seed=self.seed,
            metadata=extra_metadata or {},
        )

        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.to_json(indent=2))

        return manifest

    def _partition_splits(
        self, samples: List[SyntheticPromptSample], split_ratios: Dict[str, float]
    ) -> Dict[str, List[SyntheticPromptSample]]:
        """Partition samples into split buckets based on ratios."""
        total_ratio = sum(split_ratios.values())
        if total_ratio <= 0:
            raise ValueError("Sum of split ratios must be > 0.")

        norm_ratios = {k: v / total_ratio for k, v in split_ratios.items()}

        total_samples = len(samples)
        result: Dict[str, List[SyntheticPromptSample]] = {k: [] for k in norm_ratios.keys()}

        curr = 0
        split_keys = list(norm_ratios.keys())
        for idx, key in enumerate(split_keys):
            if idx == len(split_keys) - 1:
                # Give remaining samples to final split
                result[key] = samples[curr:]
            else:
                count = int(round(total_samples * norm_ratios[key]))
                result[key] = samples[curr : curr + count]
                curr += count

        return result

    def _write_shard_file(
        self, file_path: str, samples: List[SyntheticPromptSample]
    ) -> Tuple[int, str]:
        """Write shard file in configured format and calculate file size and sha256 checksum."""
        hasher = hashlib.sha256()

        if self.format == "jsonl":
            lines = [json.dumps(s.to_dict()) + "\n" for s in samples]
            content_bytes = "".join(lines).encode("utf-8")
        else:  # json
            serialized = [s.to_dict() for s in samples]
            content_bytes = json.dumps(serialized, indent=2).encode("utf-8")

        with open(file_path, "wb") as f:
            f.write(content_bytes)

        hasher.update(content_bytes)
        return len(content_bytes), hasher.hexdigest()


def export_dataset(
    samples: List[SyntheticPromptSample],
    output_dir: str,
    dataset_name: str = "filesystem_rl_synthetic_dataset",
    format: str = "jsonl",
    max_samples_per_shard: int = 100,
    split_ratios: Optional[Dict[str, float]] = None,
    seed: Optional[int] = 42,
    validate: bool = True,
    strict_validation: bool = False,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> DatasetManifest:
    """
    Convenience function to validate, shard, and export synthetic prompt-policy-tree samples.

    Args:
        samples: List of SyntheticPromptSample items.
        output_dir: Directory to save sharded dataset and manifest.json.
        dataset_name: Name identifier for dataset.
        format: Format string ("jsonl" or "json").
        max_samples_per_shard: Max samples per shard file.
        split_ratios: Split ratios dictionary (e.g. {"train": 0.8, "val": 0.1, "test": 0.1}).
        seed: Random seed integer.
        validate: Whether to validate samples before export.
        strict_validation: If True, raise exception on invalid sample.
        extra_metadata: Extra metadata dictionary for manifest.

    Returns:
        DatasetManifest instance.
    """
    exporter = DatasetExporter(
        output_dir=output_dir,
        dataset_name=dataset_name,
        format=format,
        max_samples_per_shard=max_samples_per_shard,
        split_ratios=split_ratios,
        seed=seed,
        validate=validate,
        strict_validation=strict_validation,
    )
    return exporter.export(samples, extra_metadata=extra_metadata)


def load_sharded_dataset(
    output_dir: str,
    split: Optional[str] = "train",
) -> List[SyntheticPromptSample]:
    """
    Load sharded dataset samples from an exported dataset directory.

    Args:
        output_dir: Path to directory containing manifest.json and split subdirectories.
        split: Specific split to load ("train", "val", "test") or None to load all splits.

    Returns:
        List of deserialized SyntheticPromptSample instances.
    """
    manifest_path = os.path.join(output_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise DatasetExportError(f"Dataset manifest not found at '{manifest_path}'.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = DatasetManifest.from_json(f.read())

    splits_to_load = [split] if split else list(manifest.splits.keys())
    loaded_samples: List[SyntheticPromptSample] = []

    for s_name in splits_to_load:
        if s_name not in manifest.splits:
            if split:
                raise DatasetExportError(
                    f"Requested split '{s_name}' not found in manifest splits ({list(manifest.splits.keys())})."
                )
            continue

        split_info = manifest.splits[s_name]
        for shard_meta in split_info.shards:
            shard_abs_path = os.path.join(output_dir, shard_meta.file_path)
            if not os.path.exists(shard_abs_path):
                raise DatasetExportError(f"Shard file not found at '{shard_abs_path}'.")

            # Verify checksum if desired
            with open(shard_abs_path, "rb") as sf:
                content = sf.read()
                checksum = hashlib.sha256(content).hexdigest()
                if checksum != shard_meta.checksum_sha256:
                    raise DatasetExportError(
                        f"Checksum mismatch for shard '{shard_abs_path}'. Expected {shard_meta.checksum_sha256}, got {checksum}."
                    )

            if manifest.format == "jsonl":
                lines = content.decode("utf-8").strip().split("\n")
                for line in lines:
                    if line.strip():
                        item_dict = json.loads(line)
                        loaded_samples.append(SyntheticPromptSample.from_dict(item_dict))
            else:  # json
                items_list = json.loads(content.decode("utf-8"))
                for item_dict in items_list:
                    loaded_samples.append(SyntheticPromptSample.from_dict(item_dict))

    return loaded_samples


def validate_dataset(
    samples: List[SyntheticPromptSample],
    strict: bool = False,
) -> Tuple[List[SyntheticPromptSample], List[ValidationResult]]:
    """
    Validate synthetic prompt-policy-tree samples using DatasetValidator.

    Args:
        samples: List of SyntheticPromptSample items.
        strict: If True, raises DatasetValidationError on any validation failure.

    Returns:
        Tuple of (valid_samples, validation_results).
    """
    validator = DatasetValidator()
    return validator.validate_batch(samples, strict=strict)
