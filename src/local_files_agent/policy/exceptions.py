"""Policy exceptions for local-files-agent."""


class PolicyError(Exception):
    """Base exception for policy related errors."""

    pass


class PolicyValidationError(PolicyError, ValueError):
    """Exception raised when a policy schema or constraint validation fails."""

    pass


class PolicyViolationError(PolicyError):
    """Exception raised when an agent action violates defined policy rules."""

    pass
