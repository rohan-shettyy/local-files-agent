"""Exceptions for noise tree and dataset generator."""

class GeneratorError(Exception):
    """Base exception for filesystem tree generation errors."""
    pass


class TreeGenerationError(GeneratorError):
    """Raised when virtual filesystem tree generation fails."""
    pass


class DatasetValidationError(GeneratorError):
    """Raised when synthetic dataset validation fails."""
    pass


class DatasetExportError(GeneratorError):
    """Raised when dataset sharding or export fails."""
    pass

