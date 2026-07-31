"""Convenience prompt generator functions bridging generator module and teacher pipeline."""

from typing import Any, Dict, List, Optional

from local_files_agent.generator.config import NoiseTreeConfig, UnorganizedTreeOutput
from local_files_agent.policy.models import PolicyConfig
from local_files_agent.teacher.models import SyntheticPromptSample
from local_files_agent.teacher.pipeline import TeacherPromptPipeline
from local_files_agent.teacher.providers import BaseTeacherProvider


def generate_synthetic_prompt(
    policy: Optional[PolicyConfig] = None,
    unorganized_tree: Optional[UnorganizedTreeOutput] = None,
    persona: str = "casual",
    provider: Optional[BaseTeacherProvider] = None,
    seed: Optional[int] = None,
) -> SyntheticPromptSample:
    """
    Generate a single synthetic prompt sample pairing user prompt, policy, and tree output.

    Args:
        policy: Optional PolicyConfig.
        unorganized_tree: Optional UnorganizedTreeOutput.
        persona: Desired persona style ("casual", "terse", "detailed", "imperfect", "urgent").
        provider: Optional BaseTeacherProvider instance.
        seed: Optional integer seed for reproducibility.

    Returns:
        SyntheticPromptSample instance.
    """
    pipeline = TeacherPromptPipeline(provider=provider)
    return pipeline.generate_sample(
        policy=policy,
        unorganized_tree=unorganized_tree,
        persona=persona,
        seed=seed,
    )


def generate_dataset_batch(
    num_samples: int = 5,
    policies: Optional[List[PolicyConfig]] = None,
    personas: Optional[List[str]] = None,
    config: Optional[NoiseTreeConfig] = None,
    provider: Optional[BaseTeacherProvider] = None,
    seed: Optional[int] = None,
) -> List[SyntheticPromptSample]:
    """
    Generate a batch of synthetic dataset samples.

    Args:
        num_samples: Number of dataset samples to synthesize.
        policies: Optional list of PolicyConfig instances.
        personas: Optional list of persona style strings.
        config: Optional NoiseTreeConfig.
        provider: Optional BaseTeacherProvider instance.
        seed: Optional integer seed for reproducibility.

    Returns:
        List of SyntheticPromptSample instances.
    """
    pipeline = TeacherPromptPipeline(provider=provider)
    return pipeline.batch_generate(
        num_samples=num_samples,
        policies=policies,
        personas=personas,
        config=config,
        seed=seed,
    )
