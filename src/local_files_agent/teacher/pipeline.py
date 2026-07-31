"""Dataset generation pipeline utilizing Teacher LLM for synthetic prompt synthesis."""

import json
import os
import random
from typing import Any, Dict, List, Optional

from local_files_agent.generator.config import NoiseTreeConfig, UnorganizedTreeOutput
from local_files_agent.generator.noise_tree import generate_unorganized_tree
from local_files_agent.policy.models import PolicyConfig
from local_files_agent.teacher.models import PromptPersona, SyntheticPromptSample
from local_files_agent.teacher.providers import (
    BaseTeacherProvider,
    GeminiTeacherProvider,
    TemplateTeacherProvider,
)


class TeacherPromptPipeline:
    """
    High-level synthetic dataset generation pipeline.
    Combines Policy JSON generation, virtual unorganized tree construction, and Teacher LLM synthesis.
    """

    def __init__(
        self,
        provider: Optional[BaseTeacherProvider] = None,
        auto_fallback: bool = True,
    ):
        """
        Initialize TeacherPromptPipeline.

        Args:
            provider: Explicit teacher provider instance (GeminiTeacherProvider, TemplateTeacherProvider, etc.).
            auto_fallback: Automatically fallback to TemplateTeacherProvider if Gemini API key missing.
        """
        if provider is not None:
            self.provider = provider
        else:
            # Auto-detect Gemini key or use Template fallback
            has_gemini_key = bool(
                os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            )
            if has_gemini_key:
                self.provider = GeminiTeacherProvider()
            elif auto_fallback:
                self.provider = TemplateTeacherProvider()
            else:
                self.provider = GeminiTeacherProvider()

    def generate_sample(
        self,
        policy: Optional[PolicyConfig] = None,
        unorganized_tree: Optional[UnorganizedTreeOutput] = None,
        persona: str = "casual",
        seed: Optional[int] = None,
    ) -> SyntheticPromptSample:
        """
        Synthesize a single dataset item pairing policy, unorganized tree, persona, and user prompt.

        Args:
            policy: Optional PolicyConfig constraints. Auto-generated if None.
            unorganized_tree: Optional UnorganizedTreeOutput. Generated if None.
            persona: Persona style for prompt synthesis.
            seed: Optional integer seed for reproducible generation.

        Returns:
            SyntheticPromptSample output item.
        """
        # If tree is provided but policy isn't, extract policy from tree
        if unorganized_tree is not None and policy is None:
            active_policy = unorganized_tree.policy
            active_tree = unorganized_tree
        elif unorganized_tree is None:
            # Generate unorganized tree and policy together
            active_tree = generate_unorganized_tree(policy=policy, seed=seed)
            active_policy = active_tree.policy
        else:
            active_policy = policy or unorganized_tree.policy
            active_tree = unorganized_tree

        # Synthesize prompt
        prompt_text = self.provider.generate_prompt(
            policy=active_policy,
            unorganized_tree=active_tree,
            persona=persona,
        )

        metadata = {
            "provider": self.provider.__class__.__name__,
            "persona": persona,
            "seed": seed,
            "target_count": len(active_tree.target_files),
            "noise_count": len(active_tree.noise_files),
            "forbidden_count": len(active_tree.forbidden_files),
        }

        return SyntheticPromptSample(
            prompt=prompt_text,
            policy=active_policy,
            persona=persona,
            unorganized_tree_output=active_tree,
            metadata=metadata,
        )

    def batch_generate(
        self,
        num_samples: int = 5,
        policies: Optional[List[PolicyConfig]] = None,
        personas: Optional[List[str]] = None,
        config: Optional[NoiseTreeConfig] = None,
        seed: Optional[int] = None,
    ) -> List[SyntheticPromptSample]:
        """
        Generate a dataset batch of synthetic prompt samples.

        Args:
            num_samples: Number of dataset samples to synthesize.
            policies: Optional list of PolicyConfig instances to cycle through.
            personas: Optional list of persona style names to cycle through.
            config: Optional NoiseTreeConfig for unorganized tree generation.
            seed: Optional seed for deterministic dataset generation.

        Returns:
            List of SyntheticPromptSample items.
        """
        rng = random.Random(seed)
        samples: List[SyntheticPromptSample] = []
        available_personas = personas or [p.value for p in PromptPersona]

        for i in range(num_samples):
            sample_seed = rng.randint(1000, 999999) if seed is not None else None
            
            pol = None
            if policies:
                pol = policies[i % len(policies)]

            persona = available_personas[i % len(available_personas)]

            tree_output = generate_unorganized_tree(policy=pol, config=config, seed=sample_seed)

            sample = self.generate_sample(
                policy=tree_output.policy,
                unorganized_tree=tree_output,
                persona=persona,
                seed=sample_seed,
            )
            samples.append(sample)

        return samples

    @staticmethod
    def save_dataset_json(
        samples: List[SyntheticPromptSample],
        file_path: str,
        indent: int = 2,
    ) -> None:
        """Save a list of SyntheticPromptSample items to a JSON dataset file."""
        serialized = [sample.to_dict() for sample in samples]
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=indent)

    @staticmethod
    def load_dataset_json(file_path: str) -> List[SyntheticPromptSample]:
        """Load a list of SyntheticPromptSample items from a JSON dataset file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError(f"Dataset JSON at {file_path} must contain a list of samples.")
            
        return [SyntheticPromptSample.from_dict(item) for item in data]
