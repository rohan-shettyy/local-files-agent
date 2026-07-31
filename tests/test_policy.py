"""Unit tests for Policy Schema Definition, Validator, and Action Compliance."""

import json
import pytest
from local_files_agent.policy import (
    PolicyConfig,
    PolicyError,
    PolicyValidationError,
    PolicyValidator,
    PolicyViolationError,
    check_action,
    validate_action,
    validate_policy_dict,
    validate_policy_json,
)


def test_default_policy_config():
    policy = PolicyConfig()
    assert policy.allow_delete is False
    assert policy.allowed_root == "Downloads"
    assert policy.target_folders == []
    assert policy.category_rules == {}
    assert policy.forbidden_paths == []
    assert policy.max_moves is None


def test_policy_config_example_from_spec():
    raw_data = {
        "allow_delete": False,
        "allowed_root": "Downloads",
        "target_folders": ["Receipts", "Screenshots", "Installers", "Notes"],
        "category_rules": {
            "Receipts": ["invoice", ".pdf"],
            "Screenshots": ["IMG_", "Screenshot"],
            "Installers": [".dmg", ".pkg", ".exe", ".msi"],
            "Notes": [".txt", ".md"],
        },
    }
    policy = PolicyConfig.from_dict(raw_data)
    assert policy.allow_delete is False
    assert policy.allowed_root == "Downloads"
    assert len(policy.target_folders) == 4
    assert policy.category_rules["Receipts"] == ["invoice", ".pdf"]


def test_json_serialization_deserialization():
    policy = PolicyConfig(
        allow_delete=True,
        allowed_root="Downloads",
        target_folders=["Docs", "Images"],
        category_rules={"Docs": [".pdf", ".docx"], "Images": [".png", ".jpg"]},
        forbidden_paths=["Downloads/sys"],
        max_moves=10,
    )
    json_str = policy.to_json()
    reconstructed = PolicyConfig.from_json(json_str)

    assert reconstructed.allow_delete is True
    assert reconstructed.allowed_root == "Downloads"
    assert reconstructed.target_folders == ["Docs", "Images"]
    assert reconstructed.category_rules == {"Docs": [".pdf", ".docx"], "Images": [".png", ".jpg"]}
    assert reconstructed.forbidden_paths == ["Downloads/sys"]
    assert reconstructed.max_moves == 10


def test_schema_export():
    schema = PolicyConfig.get_json_schema()
    assert "properties" in schema
    assert "allow_delete" in schema["properties"]
    assert "allowed_root" in schema["properties"]
    assert "target_folders" in schema["properties"]
    assert "category_rules" in schema["properties"]


def test_validation_errors():
    with pytest.raises(PolicyValidationError):
        PolicyConfig.from_dict({"allowed_root": ""})

    with pytest.raises(PolicyValidationError):
        PolicyConfig.from_dict({"target_folders": [""]})

    with pytest.raises(PolicyValidationError):
        PolicyConfig.from_dict({"category_rules": {"": ["rule"]}})

    with pytest.raises(PolicyValidationError):
        PolicyConfig.from_json("invalid json {")


def test_sync_target_folders_with_categories():
    raw = {
        "category_rules": {
            "Invoices": ["invoice"],
            "Logs": [".log"],
        }
    }
    policy = PolicyConfig.from_dict(raw)
    assert "Invoices" in policy.target_folders
    assert "Logs" in policy.target_folders


def test_get_category_for_file():
    policy = PolicyConfig(
        category_rules={
            "Receipts": ["invoice", ".pdf"],
            "Screenshots": ["IMG_", "Screenshot"],
            "Installers": [".dmg", ".pkg"],
            "Notes": [".txt", ".md"],
        }
    )

    assert policy.get_category_for_file("my_invoice_2026.png") == "Receipts"
    assert policy.get_category_for_file("statement.PDF") == "Receipts"
    assert policy.get_category_for_file("IMG_1024.PNG") == "Screenshots"
    assert policy.get_category_for_file("app_installer.dmg") == "Installers"
    assert policy.get_category_for_file("todo.md") == "Notes"
    assert policy.get_category_for_file("random_file.xyz") is None


def test_is_path_allowed():
    policy = PolicyConfig(
        allowed_root="Downloads",
        forbidden_paths=["Downloads/Protected"],
    )

    assert policy.is_path_allowed("Downloads/Receipts/invoice.pdf") is True
    assert policy.is_path_allowed("/Downloads/Notes/readme.txt") is True
    assert policy.is_path_allowed("Documents/secret.txt") is False
    assert policy.is_path_allowed("/etc/passwd") is False
    assert policy.is_path_allowed("Downloads/Protected/sys.config") is False


def test_validator_check_and_validate_action():
    policy = PolicyConfig(
        allow_delete=False,
        allowed_root="Downloads",
    )

    # Valid read/move within allowed_root
    ok, err = check_action(policy, "read", "Downloads/file.txt")
    assert ok is True
    assert err is None
    validate_action(policy, "move", "Downloads/file.txt", destination_path="Downloads/Receipts/file.txt")

    # Forbidden delete
    ok, err = check_action(policy, "delete", "Downloads/file.txt")
    assert ok is False
    assert "prohibited by policy" in err
    with pytest.raises(PolicyViolationError) as exc_info:
        validate_action(policy, "delete", "Downloads/file.txt")
    assert "prohibited by policy" in str(exc_info.value)

    # Forbidden path outside root
    with pytest.raises(PolicyViolationError) as exc_info:
        validate_action(policy, "move", "Downloads/file.txt", destination_path="Desktop/file.txt")
    assert "outside allowed root" in str(exc_info.value)


def test_validator_helper_functions():
    policy_dict = {
        "allow_delete": True,
        "allowed_root": "Downloads",
        "target_folders": ["Receipts"],
        "category_rules": {"Receipts": [".pdf"]},
    }
    policy = validate_policy_dict(policy_dict)
    assert policy.allow_delete is True

    json_str = policy.to_json()
    policy_from_json = validate_policy_json(json_str)
    assert policy_from_json.allowed_root == "Downloads"
