"""Exceptions for Teacher LLM prompt generation pipeline."""

from local_files_agent.generator.exceptions import GeneratorError


class TeacherError(GeneratorError):
    """Base exception for all Teacher LLM operations."""
    pass


class TeacherAPIError(TeacherError):
    """Raised when Teacher LLM API call fails or receives an invalid response."""
    pass


class PromptGenerationError(TeacherError):
    """Raised when prompt synthesis or parsing fails."""
    pass
