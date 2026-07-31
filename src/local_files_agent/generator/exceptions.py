"""Exceptions for noise tree and dataset generator."""

class GeneratorError(Exception):
    """Base exception for filesystem tree generation errors."""
    pass


class TreeGenerationError(GeneratorError):
    """Raised when virtual filesystem tree generation fails."""
    pass
