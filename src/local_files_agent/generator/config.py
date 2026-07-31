"""Configuration schema and metadata models for unorganized filesystem tree generation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

from local_files_agent.policy.models import PolicyConfig
from local_files_agent.virtual_fs.models import VirtualTree


class NoiseTreeConfig(BaseModel):
    """
    Configuration options for generating noisy unorganized virtual filesystem trees.

    Attributes:
        allowed_root: Root directory where unorganized files are generated.
        min_target_files: Minimum number of policy-relevant mislocated files to generate.
        max_target_files: Maximum number of policy-relevant mislocated files to generate.
        min_noise_files: Minimum number of random non-category noise files to generate.
        max_noise_files: Maximum number of random non-category noise files to generate.
        min_noise_dirs: Minimum number of unorganized subdirectories to generate.
        max_noise_dirs: Maximum number of unorganized subdirectories to generate.
        include_forbidden_paths: Whether to populate system/forbidden files defined by policy.
        max_nesting_depth: Maximum directory nesting depth for unorganized file placement.
        seed: Random seed for deterministic reproducible generation.
    """

    allowed_root: str = Field(
        default="Downloads",
        description="Root directory where unorganized files are generated.",
    )
    min_target_files: int = Field(
        default=5,
        ge=1,
        description="Minimum number of target-relevant mislocated files to generate.",
    )
    max_target_files: int = Field(
        default=15,
        ge=1,
        description="Maximum number of target-relevant mislocated files to generate.",
    )
    min_noise_files: int = Field(
        default=3,
        ge=0,
        description="Minimum number of random non-category noise files to generate.",
    )
    max_noise_files: int = Field(
        default=10,
        ge=0,
        description="Maximum number of random non-category noise files to generate.",
    )
    min_noise_dirs: int = Field(
        default=1,
        ge=0,
        description="Minimum number of unorganized noise subdirectories to generate.",
    )
    max_noise_dirs: int = Field(
        default=3,
        ge=0,
        description="Maximum number of unorganized noise subdirectories to generate.",
    )
    include_forbidden_paths: bool = Field(
        default=True,
        description="Whether to generate forbidden system files specified in policy.",
    )
    max_nesting_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum nested folder depth for unorganized file placement.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional seed for deterministic generation.",
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "NoiseTreeConfig":
        """Verify min bounds do not exceed max bounds."""
        if self.min_target_files > self.max_target_files:
            raise ValueError(
                f"min_target_files ({self.min_target_files}) cannot exceed max_target_files ({self.max_target_files})."
            )
        if self.min_noise_files > self.max_noise_files:
            raise ValueError(
                f"min_noise_files ({self.min_noise_files}) cannot exceed max_noise_files ({self.max_noise_files})."
            )
        if self.min_noise_dirs > self.max_noise_dirs:
            raise ValueError(
                f"min_noise_dirs ({self.min_noise_dirs}) cannot exceed max_noise_dirs ({self.max_noise_dirs})."
            )
        return self


class TargetFileInfo(BaseModel):
    """Metadata tracking a mislocated file within the generated environment."""

    filename: str
    current_path: str
    expected_category: str
    expected_target_dir: str
    expected_path: str


class UnorganizedTreeOutput(BaseModel):
    """Output bundle containing the generated virtual filesystem tree and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tree: VirtualTree
    policy: PolicyConfig
    target_files: List[TargetFileInfo] = Field(default_factory=list)
    noise_files: List[str] = Field(default_factory=list)
    forbidden_files: List[str] = Field(default_factory=list)
