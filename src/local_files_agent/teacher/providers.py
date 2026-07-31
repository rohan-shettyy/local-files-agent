"""Teacher LLM providers for synthesizing natural language filesystem prompts."""

import json
import os
import random
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from local_files_agent.generator.config import UnorganizedTreeOutput
from local_files_agent.policy.models import PolicyConfig
from local_files_agent.teacher.exceptions import (
    PromptGenerationError,
    TeacherAPIError,
)
from local_files_agent.teacher.models import PromptPersona, TeacherConfig

TEACHER_SYSTEM_PROMPT = (
    "You are a synthetic dataset generator for training an AI Virtual Filesystem Agent.\n"
    "Your task is to convert a given filesystem organization policy JSON (and optional tree structure)\n"
    "into a realistic natural language user request asking an assistant to organize their files.\n\n"
    "Personas:\n"
    "- casual: Friendly, conversational request (e.g. 'Hey, can you clean up my Downloads folder? Put receipts into Receipts...').\n"
    "- terse: Direct, brief instruction list (e.g. 'Organize Downloads. Move invoices/PDFs to Receipts...').\n"
    "- detailed: Comprehensive, step-by-step request explaining folder destinations and safety rules.\n"
    "- imperfect: Informal user request with slight typos or informal file descriptions.\n"
    "- urgent: Time-sensitive request asking for immediate cleanup.\n\n"
    "Rules:\n"
    "1. Do NOT output markdown code blocks or JSON. Output ONLY the natural language prompt text.\n"
    "2. Ensure the prompt explicitly reflects the allowed root directory and category rules.\n"
    "3. Keep the prompt natural, realistic, and clear."
)


class BaseTeacherProvider(ABC):
    """Abstract base class for Teacher LLM prompt synthesis providers."""

    @abstractmethod
    def generate_prompt(
        self,
        policy: PolicyConfig,
        unorganized_tree: Optional[UnorganizedTreeOutput] = None,
        persona: str = "casual",
    ) -> str:
        """
        Synthesize a natural language prompt string for a given policy and persona.

        Args:
            policy: PolicyConfig specifying filesystem constraints and rules.
            unorganized_tree: Optional UnorganizedTreeOutput generated tree metadata.
            persona: Persona style identifier (casual, terse, detailed, imperfect, urgent).

        Returns:
            Synthesized user prompt string.
        """
        pass


class GeminiTeacherProvider(BaseTeacherProvider):
    """
    Teacher LLM provider integrating with Google AI Studio / Gemini API (e.g. gemini-2.5-flash).
    Uses free tier API keys supplied directly or via GEMINI_API_KEY / GOOGLE_API_KEY environment variables.
    """

    def __init__(
        self,
        config: Optional[TeacherConfig] = None,
        api_key: Optional[str] = None,
    ):
        self.config = config or TeacherConfig(provider="gemini")
        self.api_key = (
            api_key
            or self.config.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

    def generate_prompt(
        self,
        policy: PolicyConfig,
        unorganized_tree: Optional[UnorganizedTreeOutput] = None,
        persona: str = "casual",
    ) -> str:
        """Invoke Gemini API to synthesize prompt text based on policy JSON."""
        if not self.api_key:
            raise TeacherAPIError(
                "Gemini API key is required. Pass api_key parameter or set GEMINI_API_KEY / GOOGLE_API_KEY."
            )

        model = self.config.model_name
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"

        tree_summary = ""
        if unorganized_tree:
            targets = [tf.filename for tf in unorganized_tree.target_files]
            tree_summary = f"\nUnorganized files present: {targets[:8]}"

        user_content = (
            f"Policy JSON:\n{policy.to_json()}\n"
            f"Desired Persona: {persona}{tree_summary}\n\n"
            "Please generate the natural language user instruction:"
        )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{TEACHER_SYSTEM_PROMPT}\n\n{user_content}"}],
                }
            ],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                candidates = resp_data.get("candidates", [])
                if not candidates:
                    raise TeacherAPIError("No response candidates returned by Gemini API.")
                
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts or "text" not in parts[0]:
                    raise TeacherAPIError("Malformed candidate format in Gemini API response.")
                
                prompt_text = parts[0]["text"].strip()
                if not prompt_text:
                    raise PromptGenerationError("Gemini API returned an empty prompt string.")
                return prompt_text

        except urllib.error.HTTPError as err:
            body = err.read().decode("utf-8") if err.fp else str(err)
            raise TeacherAPIError(f"Gemini API HTTP Error {err.code}: {body}") from err
        except urllib.error.URLError as err:
            raise TeacherAPIError(f"Gemini API connection error: {err.reason}") from err
        except json.JSONDecodeError as err:
            raise TeacherAPIError(f"Failed to parse Gemini API JSON response: {err}") from err
        except Exception as err:
            if isinstance(err, (TeacherAPIError, PromptGenerationError)):
                raise
            raise TeacherAPIError(f"Unexpected error during Gemini API invocation: {err}") from err


class TemplateTeacherProvider(BaseTeacherProvider):
    """
    Deterministic rule & template-based prompt synthesizer.
    Generates rich, realistic natural language filesystem prompts without external API dependencies.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def generate_prompt(
        self,
        policy: PolicyConfig,
        unorganized_tree: Optional[UnorganizedTreeOutput] = None,
        persona: str = "casual",
    ) -> str:
        """Synthesize prompt string using rule-based template logic."""
        rng = random.Random(self.seed)
        root = policy.allowed_root

        # Format category rules description
        rule_descriptions: List[str] = []
        for cat, rules in policy.category_rules.items():
            rule_str = ", ".join(rules)
            rule_descriptions.append(f"files matching ({rule_str}) into '{cat}'")

        if not rule_descriptions and policy.target_folders:
            for cat in policy.target_folders:
                rule_descriptions.append(f"relevant files into '{cat}'")

        rules_summary = "; ".join(rule_descriptions) if rule_descriptions else "mislocated files into target folders"

        p_norm = str(persona).lower()
        if p_norm == PromptPersona.TERSE.value:
            prefix = rng.choice([
                f"Organize '{root}' directory.",
                f"Clean up '{root}'.",
                f"Sort files in '{root}'.",
            ])
            prompt = f"{prefix} Move {rules_summary}."

        elif p_norm == PromptPersona.DETAILED.value:
            forbidden_msg = ""
            if policy.forbidden_paths:
                forbidden_msg = f" Do not touch or delete protected system files like {policy.forbidden_paths[0]}."
            delete_msg = " Safe to delete redundant junk files." if policy.allow_delete else " Do not delete any files."
            prompt = (
                f"Please review the unorganized files in '{root}'. "
                f"Your goal is to organize the directory by moving {rules_summary}.{delete_msg}{forbidden_msg}"
            )

        elif p_norm == PromptPersona.IMPERFECT.value:
            prefix = rng.choice([
                f"hey can u sort out my {root} folder?",
                f"my {root} dir is super messy pls fix it,",
                f"can u clean {root} for me,",
            ])
            prompt = f"{prefix} put {rules_summary}. thanks!"

        elif p_norm == PromptPersona.URGENT.value:
            prompt = (
                f"Quick! My '{root}' directory is completely cluttered. "
                f"Please organize it right away by sorting {rules_summary} ASAP."
            )

        else:  # CASUAL or NATURAL default
            prefix = rng.choice([
                f"Hey! Could you help me organize my '{root}' directory?",
                f"Hi! My '{root}' folder is getting pretty cluttered.",
                f"Hello! I need help sorting out files in my '{root}' directory.",
            ])
            prompt = f"{prefix} Please move {rules_summary}."

        return prompt.strip()


class MockTeacherProvider(BaseTeacherProvider):
    """Mock teacher provider for testing pipeline integrations."""

    def __init__(self, response_text: Optional[str] = None):
        self.response_text = response_text

    def generate_prompt(
        self,
        policy: PolicyConfig,
        unorganized_tree: Optional[UnorganizedTreeOutput] = None,
        persona: str = "casual",
    ) -> str:
        """Return preset or default mock response string."""
        if self.response_text is not None:
            return self.response_text
        return f"Mock prompt for root '{policy.allowed_root}' with persona '{persona}'."
