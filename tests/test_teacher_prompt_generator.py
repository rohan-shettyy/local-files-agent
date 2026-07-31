"""Unit tests for Teacher LLM prompt generation pipeline."""

import json
import os
import tempfile
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from local_files_agent.generator import (
    NoiseTreeConfig,
    generate_dataset_batch,
    generate_synthetic_prompt,
    generate_unorganized_tree,
)
from local_files_agent.policy.models import PolicyConfig
from local_files_agent.teacher import (
    BaseTeacherProvider,
    GeminiTeacherProvider,
    MockTeacherProvider,
    PromptGenerationError,
    PromptPersona,
    SyntheticPromptSample,
    TeacherAPIError,
    TeacherConfig,
    TeacherError,
    TeacherPromptPipeline,
    TemplateTeacherProvider,
)


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


class TestTeacherModels:
    def test_teacher_config_defaults(self):
        config = TeacherConfig()
        assert config.provider == "gemini"
        assert config.model_name == "gemini-2.5-flash"
        assert config.temperature == 0.7
        assert config.max_tokens == 500
        assert config.timeout_seconds == 30.0

    def test_synthetic_prompt_sample_serialization(self, sample_policy):
        tree_output = generate_unorganized_tree(policy=sample_policy, seed=42)
        sample = SyntheticPromptSample(
            prompt="Organize my Downloads folder.",
            policy=sample_policy,
            persona="casual",
            unorganized_tree_output=tree_output,
            metadata={"seed": 42},
        )

        d = sample.to_dict()
        assert d["prompt"] == "Organize my Downloads folder."
        assert d["persona"] == "casual"
        assert "policy" in d
        assert "target_files" in d

        json_str = sample.to_json()
        restored = SyntheticPromptSample.from_json(json_str)
        assert restored.prompt == sample.prompt
        assert restored.persona == sample.persona
        assert restored.policy.allowed_root == sample_policy.allowed_root


class TestProviders:
    def test_mock_teacher_provider(self, sample_policy):
        provider = MockTeacherProvider(response_text="Custom mock prompt.")
        prompt = provider.generate_prompt(policy=sample_policy)
        assert prompt == "Custom mock prompt."

        default_provider = MockTeacherProvider()
        prompt_def = default_provider.generate_prompt(policy=sample_policy)
        assert "Downloads" in prompt_def

    def test_template_teacher_provider_personas(self, sample_policy):
        provider = TemplateTeacherProvider(seed=123)

        for persona in [PromptPersona.CASUAL, PromptPersona.TERSE, PromptPersona.DETAILED, PromptPersona.IMPERFECT, PromptPersona.URGENT]:
            prompt = provider.generate_prompt(policy=sample_policy, persona=persona.value)
            assert isinstance(prompt, str)
            assert len(prompt) > 10
            assert "Downloads" in prompt or "downloads" in prompt

    def test_gemini_provider_missing_key(self, sample_policy):
        with patch.dict(os.environ, {}, clear=True):
            provider = GeminiTeacherProvider(api_key=None)
            with pytest.raises(TeacherAPIError, match="Gemini API key is required"):
                provider.generate_prompt(policy=sample_policy)

    @patch("urllib.request.urlopen")
    def test_gemini_provider_success(self, mock_urlopen, sample_policy):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Hey assistant, please organize my Downloads folder into Receipts and Screenshots."}
                        ]
                    }
                }
            ]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        provider = GeminiTeacherProvider(api_key="fake-test-key")
        prompt = provider.generate_prompt(policy=sample_policy, persona="casual")

        assert prompt == "Hey assistant, please organize my Downloads folder into Receipts and Screenshots."
        assert mock_urlopen.called

    @patch("urllib.request.urlopen")
    def test_gemini_provider_http_error(self, mock_urlopen, sample_policy):
        mock_err = urllib.error.HTTPError(
            url="http://fake", code=401, msg="Unauthorized", hdrs={}, fp=None
        )
        mock_urlopen.side_effect = mock_err

        provider = GeminiTeacherProvider(api_key="invalid-key")
        with pytest.raises(TeacherAPIError, match="Gemini API HTTP Error 401"):
            provider.generate_prompt(policy=sample_policy)


class TestPipeline:
    def test_pipeline_auto_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            pipeline = TeacherPromptPipeline(auto_fallback=True)
            assert isinstance(pipeline.provider, TemplateTeacherProvider)

    def test_generate_sample(self, sample_policy):
        provider = TemplateTeacherProvider(seed=42)
        pipeline = TeacherPromptPipeline(provider=provider)

        sample = pipeline.generate_sample(policy=sample_policy, persona="terse", seed=42)
        assert isinstance(sample, SyntheticPromptSample)
        assert "Downloads" in sample.prompt
        assert sample.persona == "terse"
        assert sample.unorganized_tree_output is not None

    def test_batch_generate(self, sample_policy):
        provider = TemplateTeacherProvider(seed=99)
        pipeline = TeacherPromptPipeline(provider=provider)

        samples = pipeline.batch_generate(
            num_samples=3,
            policies=[sample_policy],
            personas=["casual", "terse"],
            seed=99,
        )

        assert len(samples) == 3
        assert all(isinstance(s, SyntheticPromptSample) for s in samples)

    def test_save_and_load_dataset_json(self, sample_policy):
        provider = TemplateTeacherProvider(seed=42)
        pipeline = TeacherPromptPipeline(provider=provider)
        samples = pipeline.batch_generate(num_samples=2, policies=[sample_policy], seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "dataset.json")
            pipeline.save_dataset_json(samples, file_path)

            assert os.path.exists(file_path)
            loaded_samples = pipeline.load_dataset_json(file_path)

            assert len(loaded_samples) == 2
            assert loaded_samples[0].prompt == samples[0].prompt
            assert loaded_samples[0].policy.allowed_root == samples[0].policy.allowed_root


class TestConvenienceFunctions:
    def test_generate_synthetic_prompt(self, sample_policy):
        provider = MockTeacherProvider("Synthetic test prompt")
        sample = generate_synthetic_prompt(policy=sample_policy, provider=provider)
        assert sample.prompt == "Synthetic test prompt"

    def test_generate_dataset_batch(self, sample_policy):
        provider = TemplateTeacherProvider(seed=77)
        batch = generate_dataset_batch(num_samples=3, policies=[sample_policy], provider=provider, seed=77)
        assert len(batch) == 3
