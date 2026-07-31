"""Data models and configurations for Teacher LLM prompt generation."""

import json
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from local_files_agent.generator.config import UnorganizedTreeOutput
from local_files_agent.policy.models import PolicyConfig


class PromptPersona(str, Enum):
    """Supported persona styles for synthesizing natural language prompts."""

    CASUAL = "casual"
    TERSE = "terse"
    DETAILED = "detailed"
    IMPERFECT = "imperfect"
    URGENT = "urgent"
    NATURAL = "natural"


class TeacherConfig(BaseModel):
    """
    Configuration options for Teacher LLM prompt generation providers.

    Attributes:
        provider: Provider identifier ("gemini", "template", "mock").
        model_name: Model identifier for API providers (e.g., "gemini-2.5-flash").
        api_key: Optional API key for external LLM service.
        temperature: Sampling temperature for text generation.
        max_tokens: Maximum token length for response.
        timeout_seconds: Request timeout duration in seconds.
    """

    provider: str = Field(default="gemini", description="Teacher provider backend name.")
    model_name: str = Field(
        default="gemini-2.5-flash",
        description="Teacher LLM model name (e.g. gemini-2.5-flash or gemini-1.5-flash).",
    )
    api_key: Optional[str] = Field(default=None, description="API key for teacher LLM service.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: int = Field(default=500, ge=10, le=4096, description="Max token output limit.")
    timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=300.0, description="HTTP request timeout in seconds."
    )


class SyntheticPromptSample(BaseModel):
    """
    Output data item representing a synthesized natural language prompt paired with policy & tree.

    Attributes:
        prompt: Synthesized user prompt string.
        policy: PolicyConfig defining target rules and constraints.
        persona: Persona style used during prompt synthesis.
        unorganized_tree_output: Optional UnorganizedTreeOutput generated tree metadata.
        metadata: Additional metadata dictionary (provider used, timestamp, seed, etc.).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str = Field(description="Synthesized natural language user request.")
    policy: PolicyConfig = Field(description="Associated filesystem organization policy.")
    persona: str = Field(default="casual", description="Persona style used for prompt generation.")
    unorganized_tree_output: Optional[UnorganizedTreeOutput] = Field(
        default=None, description="Optional associated unorganized virtual filesystem output."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata dictionary tracking generation details."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sample into python dictionary."""
        data = {
            "prompt": self.prompt,
            "policy": self.policy.to_dict(),
            "persona": self.persona,
            "metadata": self.metadata,
        }
        if self.unorganized_tree_output is not None:
            data["target_files"] = [
                tf.model_dump() for tf in self.unorganized_tree_output.target_files
            ]
            data["noise_files"] = self.unorganized_tree_output.noise_files
            data["forbidden_files"] = self.unorganized_tree_output.forbidden_files
        return data

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize sample to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticPromptSample":
        """Reconstruct SyntheticPromptSample from dictionary representation."""
        policy_data = data.get("policy", {})
        policy = PolicyConfig.from_dict(policy_data) if isinstance(policy_data, dict) else policy_data

        return cls(
            prompt=data["prompt"],
            policy=policy,
            persona=data.get("persona", "casual"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SyntheticPromptSample":
        """Reconstruct SyntheticPromptSample from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
