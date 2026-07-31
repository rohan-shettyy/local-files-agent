"""Generator package for unorganized virtual filesystem trees and synthetic dataset prompts."""

from local_files_agent.generator.config import (
    NoiseTreeConfig,
    TargetFileInfo,
    UnorganizedTreeOutput,
)
from local_files_agent.generator.exceptions import (
    GeneratorError,
    TreeGenerationError,
)
from local_files_agent.generator.noise_tree import (
    NoiseTreeGenerator,
    generate_unorganized_tree,
)
from local_files_agent.generator.prompt_generator import (
    generate_dataset_batch,
    generate_synthetic_prompt,
)

__all__ = [
    "NoiseTreeConfig",
    "TargetFileInfo",
    "UnorganizedTreeOutput",
    "GeneratorError",
    "TreeGenerationError",
    "NoiseTreeGenerator",
    "generate_unorganized_tree",
    "generate_synthetic_prompt",
    "generate_dataset_batch",
]
