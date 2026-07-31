"""Teacher LLM prompt generation pipeline package."""

from local_files_agent.teacher.exceptions import (
    PromptGenerationError,
    TeacherAPIError,
    TeacherError,
)
from local_files_agent.teacher.models import (
    PromptPersona,
    SyntheticPromptSample,
    TeacherConfig,
)
from local_files_agent.teacher.pipeline import TeacherPromptPipeline
from local_files_agent.teacher.providers import (
    BaseTeacherProvider,
    GeminiTeacherProvider,
    MockTeacherProvider,
    TemplateTeacherProvider,
)

__all__ = [
    "TeacherError",
    "TeacherAPIError",
    "PromptGenerationError",
    "PromptPersona",
    "TeacherConfig",
    "SyntheticPromptSample",
    "TeacherPromptPipeline",
    "BaseTeacherProvider",
    "GeminiTeacherProvider",
    "TemplateTeacherProvider",
    "MockTeacherProvider",
]
