"""Pydantic schema and models for policy constraints in RL Fine-Tuning Environment."""

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from local_files_agent.policy.exceptions import PolicyValidationError


def resolve_path_segments(path: str) -> List[str]:
    """
    Normalize and split a path string into non-empty component names.
    Handles relative and absolute path segments, '.' and '..'.
    """
    if not isinstance(path, str):
        raise PolicyValidationError(f"Path must be a string, got {type(path)}.")

    clean_path = path.strip()
    if not clean_path or clean_path == "/":
        return []

    parts = [p for p in clean_path.split("/") if p and p != "."]
    resolved: List[str] = []
    for part in parts:
        if part == "..":
            if resolved:
                resolved.pop()
        else:
            resolved.append(part)

    return resolved


class PolicyConfig(BaseModel):
    """
    Policy constraint configuration for filesystem agent execution.

    Attributes:
        allow_delete: Whether file or directory deletion is permitted.
        allowed_root: Root directory boundary restricting agent operations.
        target_folders: Destination folder names where files should be organized.
        category_rules: Rules mapping category/folder name to list of file extensions or keywords.
        forbidden_paths: List of path patterns that must not be deleted or modified.
        max_moves: Optional maximum allowed move operations per trajectory.
    """

    allow_delete: bool = Field(
        default=False,
        description="Whether file or directory deletion is permitted by the policy.",
    )
    allowed_root: str = Field(
        default="Downloads",
        description="Root directory boundary restricting agent file operations.",
    )
    target_folders: List[str] = Field(
        default_factory=list,
        description="List of target folder names for file organization.",
    )
    category_rules: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping of category/folder names to lists of keywords or file extensions.",
    )
    forbidden_paths: List[str] = Field(
        default_factory=list,
        description="List of system/protected path patterns forbidden from deletion or modification.",
    )
    max_moves: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional maximum number of allowable move operations.",
    )

    @field_validator("allowed_root")
    @classmethod
    def validate_allowed_root(cls, value: str) -> str:
        """Validate and normalize allowed_root path string."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("allowed_root must be a non-empty string.")
        return value.strip()

    @field_validator("target_folders")
    @classmethod
    def validate_target_folders(cls, value: List[str]) -> List[str]:
        """Validate target_folders list contains non-empty strings without duplicates."""
        cleaned: List[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Target folder names must be non-empty strings.")
            folder = item.strip()
            if folder not in cleaned:
                cleaned.append(folder)
        return cleaned

    @field_validator("category_rules")
    @classmethod
    def validate_category_rules(cls, value: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate category_rules dictionary structure and entries."""
        cleaned: Dict[str, List[str]] = {}
        for category, rules in value.items():
            if not isinstance(category, str) or not category.strip():
                raise ValueError("Category rule keys must be non-empty strings.")
            cat_name = category.strip()
            if not isinstance(rules, list):
                raise ValueError(f"Rules for category '{cat_name}' must be a list of strings.")
            cleaned_rules: List[str] = []
            for r in rules:
                if not isinstance(r, str) or not r.strip():
                    raise ValueError(f"Rule pattern for category '{cat_name}' must be a non-empty string.")
                rule_str = r.strip()
                if rule_str not in cleaned_rules:
                    cleaned_rules.append(rule_str)
            cleaned[cat_name] = cleaned_rules
        return cleaned

    @model_validator(mode="after")
    def sync_target_folders_with_categories(self) -> "PolicyConfig":
        """Ensure all categories in category_rules exist in target_folders."""
        for category in self.category_rules.keys():
            if category not in self.target_folders:
                self.target_folders.append(category)
        return self

    def get_category_for_file(self, filename: str) -> Optional[str]:
        """
        Determine matching category folder for a given filename based on category_rules.

        Args:
            filename: Name or path of the file to categorize.

        Returns:
            Name of matching target category folder, or None if no rule matches.
        """
        clean_name = filename.split("/")[-1].strip().lower()
        if not clean_name:
            return None

        for category, rules in self.category_rules.items():
            for rule in rules:
                rule_lower = rule.lower()
                if rule_lower.startswith("."):
                    # Extension match
                    if clean_name.endswith(rule_lower):
                        return category
                else:
                    # Keyword substring match
                    if rule_lower in clean_name:
                        return category
        return None

    def is_path_allowed(self, path: str) -> bool:
        """
        Check if a given path is within the allowed_root boundary and not forbidden.

        Args:
            path: Target file/directory path to evaluate.

        Returns:
            True if path operation is permitted under allowed_root and forbidden_paths.
        """
        path_parts = resolve_path_segments(path)
        root_parts = resolve_path_segments(self.allowed_root)

        # Check root boundary containment
        if root_parts:
            if len(path_parts) < len(root_parts) or path_parts[: len(root_parts)] != root_parts:
                return False

        # Check forbidden paths
        for forbidden in self.forbidden_paths:
            forb_parts = resolve_path_segments(forbidden)
            if forb_parts and len(path_parts) >= len(forb_parts):
                if path_parts[: len(forb_parts)] == forb_parts:
                    return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy configuration to a python dictionary."""
        return self.model_dump()

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize policy configuration to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyConfig":
        """
        Instantiate PolicyConfig from a dictionary.

        Raises:
            PolicyValidationError: If dictionary data fails policy schema validation.
        """
        try:
            return cls.model_validate(data)
        except Exception as err:
            raise PolicyValidationError(f"Invalid policy data: {err}") from err

    @classmethod
    def from_json(cls, json_str: str) -> "PolicyConfig":
        """
        Instantiate PolicyConfig from a JSON string.

        Raises:
            PolicyValidationError: If JSON is invalid or fails schema validation.
        """
        try:
            data = json.loads(json_str)
        except Exception as err:
            raise PolicyValidationError(f"Invalid JSON string: {err}") from err
        return cls.from_dict(data)

    @classmethod
    def get_json_schema(cls) -> Dict[str, Any]:
        """Return the JSON schema definition dictionary for PolicyConfig."""
        return cls.model_json_schema()
