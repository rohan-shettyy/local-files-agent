"""Validator for policy configurations and runtime agent action adherence."""

from typing import Any, Dict, Optional, Tuple

from local_files_agent.policy.exceptions import PolicyValidationError, PolicyViolationError
from local_files_agent.policy.models import PolicyConfig


class PolicyValidator:
    """Validator class for policy schema validation and action compliance checking."""

    @staticmethod
    def validate_policy_dict(data: Dict[str, Any]) -> PolicyConfig:
        """
        Validate and parse dictionary data into a PolicyConfig.

        Args:
            data: Policy dictionary.

        Returns:
            Validated PolicyConfig instance.

        Raises:
            PolicyValidationError: If data fails policy schema validation.
        """
        if not isinstance(data, dict):
            raise PolicyValidationError(f"Expected dict for policy data, got {type(data)}.")
        return PolicyConfig.from_dict(data)

    @staticmethod
    def validate_policy_json(json_str: str) -> PolicyConfig:
        """
        Validate and parse a JSON string into a PolicyConfig.

        Args:
            json_str: JSON string.

        Returns:
            Validated PolicyConfig instance.

        Raises:
            PolicyValidationError: If JSON format or schema validation fails.
        """
        return PolicyConfig.from_json(json_str)

    @staticmethod
    def check_action(
        policy: PolicyConfig,
        action: str,
        path: str,
        destination_path: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate an agent action against a PolicyConfig without raising an exception.

        Args:
            policy: PolicyConfig instance.
            action: Action type string (e.g. 'delete', 'move', 'create', 'update', 'read').
            path: Primary target path.
            destination_path: Optional destination path for move operations.

        Returns:
            Tuple of (is_valid: bool, reason_if_invalid: Optional[str]).
        """
        action_lower = action.lower().strip()

        # Check delete permission
        if action_lower in ("delete", "remove", "rm") and not policy.allow_delete:
            return False, f"Delete operation on '{path}' is prohibited by policy constraint (allow_delete=False)."

        # Check primary path containment
        if not policy.is_path_allowed(path):
            return False, f"Path '{path}' is outside allowed root '{policy.allowed_root}' or is a forbidden path."

        # Check destination path containment for moves
        if destination_path is not None:
            if not policy.is_path_allowed(destination_path):
                return (
                    False,
                    f"Destination path '{destination_path}' is outside allowed root '{policy.allowed_root}' or is a forbidden path.",
                )

        return True, None

    @classmethod
    def validate_action(
        cls,
        policy: PolicyConfig,
        action: str,
        path: str,
        destination_path: Optional[str] = None,
    ) -> None:
        """
        Validate an agent action against policy constraints, raising PolicyViolationError on violation.

        Args:
            policy: PolicyConfig instance.
            action: Action type string.
            path: Target path.
            destination_path: Optional destination path for move operations.

        Raises:
            PolicyViolationError: If action violates policy constraints.
        """
        is_valid, reason = cls.check_action(policy, action, path, destination_path)
        if not is_valid:
            raise PolicyViolationError(reason)


# Functional alias for convenient direct imports
validate_policy_dict = PolicyValidator.validate_policy_dict
validate_policy_json = PolicyValidator.validate_policy_json
validate_action = PolicyValidator.validate_action
check_action = PolicyValidator.check_action
