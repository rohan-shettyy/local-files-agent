"""Algorithmic noise tree and unorganized filesystem generator."""

import random
from typing import Any, Dict, List, Optional, Tuple

from local_files_agent.generator.config import (
    NoiseTreeConfig,
    TargetFileInfo,
    UnorganizedTreeOutput,
)
from local_files_agent.generator.exceptions import TreeGenerationError
from local_files_agent.policy.models import PolicyConfig
from local_files_agent.virtual_fs.models import NodeType, VirtualTree

# Default template data for generating realistic filenames and file contents
DEFAULT_CATEGORY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Receipts": {
        "patterns": ["invoice", "receipt", "billing", "payment", ".pdf"],
        "stems": [
            "invoice_2024_01",
            "receipt_grocery_mar",
            "payment_confirmation_uber",
            "billing_statement_q1",
            "amazon_order_details",
            "apple_store_receipt",
            "utility_bill_electric",
        ],
        "extensions": [".pdf"],
        "contents": "[PDF Binary Data] Invoice #10492 - Amount Due: $45.00",
    },
    "Screenshots": {
        "patterns": ["Screenshot", "IMG_", "Screen Shot", ".png"],
        "stems": [
            "Screenshot 2024-05-01 at 10.15.22",
            "IMG_8492",
            "Screenshot_app_bug_ui",
            "Screen Shot 2024-03-12",
            "IMG_0023_export",
            "Screenshot_dashboard_chart",
        ],
        "extensions": [".png", ".jpg"],
        "contents": "[PNG Image Data] Resolution 1920x1080",
    },
    "Installers": {
        "patterns": ["setup", "installer", ".dmg", ".pkg", ".exe", ".msi"],
        "stems": [
            "docker_desktop_v4.2",
            "vscode_installer_mac",
            "python-3.11-setup",
            "zoom_client_latest",
            "slack_desktop_v4",
            "chrome_installer",
        ],
        "extensions": [".dmg", ".pkg", ".exe", ".msi"],
        "contents": "[Binary Executable / Package Installer Package]",
    },
    "Notes": {
        "patterns": ["notes", "todo", "draft", "meeting", ".txt", ".md"],
        "stems": [
            "meeting_notes_july",
            "todo_list_project",
            "ideas_draft_v1",
            "project_readme",
            "architecture_thoughts",
            "weekly_standup_notes",
        ],
        "extensions": [".txt", ".md"],
        "contents": "Action Items:\n- Review PRs\n- Update RL environment docs\n- Run benchmarks",
    },
}

DEFAULT_NOISE_SUBDIRS = [
    "Unsorted",
    "Old_Files",
    "Drafts",
    "Misc_Downloads",
    "Temp_Exports",
    "Incoming",
    "Archive_Junk",
]

DEFAULT_NOISE_FILE_TEMPLATES = [
    ("DS_Store_file", ".DS_Store", "Mac OS Metadata Store"),
    ("desktop_ini", "desktop.ini", "[ShellClassInfo] Folder Customization"),
    ("temp_cache", "temp_cache.tmp", "Temporary cache data dump"),
    ("system_log", "system_check_2024.log", "2024-07-31 INFO System health normal"),
    ("backup_tar", "backup_old_data.tar.gz", "[Archive Gzip Stream Data]"),
    ("csv_export", "metrics_raw_export.csv", "timestamp,metric,value\n1700000,cpu,42.1"),
    ("cache_bin", "compiled_cache.bin", "\x00\x01\x02\x03BINARY_DATA"),
    ("bak_file", "old_config.bak", "# Backup config file\nKEY=VALUE"),
    ("debug_out", "debug_output.out", "Trace log: process exited with code 0"),
]


class NoiseTreeGenerator:
    """
    Algorithmic generator that constructs noisy, unorganized virtual filesystem trees
    paired with policy constraints for reinforcement learning environments.
    """

    def __init__(self, config: Optional[NoiseTreeConfig] = None):
        self.config = config or NoiseTreeConfig()

    def generate(
        self,
        policy: Optional[PolicyConfig] = None,
        config: Optional[NoiseTreeConfig] = None,
        seed: Optional[int] = None,
    ) -> UnorganizedTreeOutput:
        """
        Generate an unorganized virtual filesystem tree output bundle.

        Args:
            policy: PolicyConfig defining target folders, category rules, and constraints.
            config: NoiseTreeConfig overriding generator settings.
            seed: Optional seed for deterministic generation.

        Returns:
            UnorganizedTreeOutput containing VirtualTree, PolicyConfig, and file metadata.
        """
        active_config = config or self.config
        eff_seed = seed if seed is not None else active_config.seed
        rng = random.Random(eff_seed)

        active_policy = policy or self._create_default_policy()
        allowed_root = active_config.allowed_root or active_policy.allowed_root

        # Initialize VirtualTree
        tree = VirtualTree()

        # Ensure allowed_root directory exists
        if allowed_root and allowed_root != "/":
            tree.create(allowed_root, node_type=NodeType.DIRECTORY, create_parents=True)

        # 1. Create target category directories inside allowed_root
        target_dirs: List[str] = []
        for folder in active_policy.target_folders:
            folder_path = f"{allowed_root}/{folder}" if allowed_root != "/" else folder
            tree.create(folder_path, node_type=NodeType.DIRECTORY, create_parents=True)
            target_dirs.append(folder_path)

        # 2. Generate noise subdirectories inside allowed_root
        num_noise_dirs = rng.randint(active_config.min_noise_dirs, active_config.max_noise_dirs)
        noise_dirs: List[str] = []
        shuffled_subdirs = list(DEFAULT_NOISE_SUBDIRS)
        rng.shuffle(shuffled_subdirs)

        for i in range(num_noise_dirs):
            subdir_name = shuffled_subdirs[i % len(shuffled_subdirs)]
            if i >= len(shuffled_subdirs):
                subdir_name = f"{subdir_name}_{i}"
            subdir_path = f"{allowed_root}/{subdir_name}" if allowed_root != "/" else subdir_name
            tree.create(subdir_path, node_type=NodeType.DIRECTORY, create_parents=True)
            noise_dirs.append(subdir_path)

        # Candidate folders for mislocating files
        mislocation_candidate_dirs = [allowed_root] + noise_dirs + target_dirs

        # 3. Generate target-relevant mislocated files
        num_target_files = rng.randint(active_config.min_target_files, active_config.max_target_files)
        target_file_infos: List[TargetFileInfo] = []
        used_paths = set()

        categories = list(active_policy.category_rules.keys())
        if not categories and active_policy.target_folders:
            categories = list(active_policy.target_folders)

        for _ in range(num_target_files):
            if not categories:
                break
            category = rng.choice(categories)
            filename, content = self._generate_filename_for_category(category, active_policy, rng)

            expected_target_dir = f"{allowed_root}/{category}" if allowed_root != "/" else category
            expected_path = f"{expected_target_dir}/{filename}"

            # Pick a mislocated directory that is NOT expected_target_dir
            wrong_dirs = [d for d in mislocation_candidate_dirs if d != expected_target_dir]
            if not wrong_dirs:
                wrong_dirs = [allowed_root]

            current_dir = rng.choice(wrong_dirs)
            current_path = f"{current_dir}/{filename}"

            # Avoid path collisions
            attempts = 0
            while current_path in used_paths and attempts < 20:
                stem, ext = (filename.rsplit(".", 1) + [""])[:2]
                ext_str = f".{ext}" if ext else ""
                filename = f"{stem}_{rng.randint(10, 99)}{ext_str}"
                current_path = f"{current_dir}/{filename}"
                expected_path = f"{expected_target_dir}/{filename}"
                attempts += 1

            used_paths.add(current_path)

            tree.create(current_path, node_type=NodeType.FILE, contents=content, create_parents=True)

            target_file_infos.append(
                TargetFileInfo(
                    filename=filename,
                    current_path=current_path,
                    expected_category=category,
                    expected_target_dir=expected_target_dir,
                    expected_path=expected_path,
                )
            )

        # 4. Generate random noise files (non-category)
        num_noise_files = rng.randint(active_config.min_noise_files, active_config.max_noise_files)
        noise_file_paths: List[str] = []

        for _ in range(num_noise_files):
            noise_name, noise_content = self._generate_noise_filename(active_policy, rng)
            chosen_dir = rng.choice(mislocation_candidate_dirs)
            current_path = f"{chosen_dir}/{noise_name}"

            attempts = 0
            while current_path in used_paths and attempts < 20:
                stem, ext = (noise_name.rsplit(".", 1) + [""])[:2]
                ext_str = f".{ext}" if ext else ""
                noise_name = f"{stem}_{rng.randint(10, 99)}{ext_str}"
                current_path = f"{chosen_dir}/{noise_name}"
                attempts += 1

            used_paths.add(current_path)
            tree.create(current_path, node_type=NodeType.FILE, contents=noise_content, create_parents=True)
            noise_file_paths.append(current_path)

        # 5. Generate forbidden system files
        forbidden_file_paths: List[str] = []
        if active_config.include_forbidden_paths and active_policy.forbidden_paths:
            for forb_pattern in active_policy.forbidden_paths:
                forb_path = self._build_forbidden_path(forb_pattern, allowed_root)
                if forb_path not in used_paths:
                    used_paths.add(forb_path)
                    try:
                        node = tree.create(
                            forb_path,
                            node_type=NodeType.FILE,
                            contents="[SYSTEM PROTECTED FILE - DO NOT DELETE OR MODIFY]",
                            create_parents=True,
                        )
                        node.metadata.read_only = True
                        forbidden_file_paths.append(forb_path)
                    except Exception as err:
                        # Directory might already exist or path conflict
                        pass

        return UnorganizedTreeOutput(
            tree=tree,
            policy=active_policy,
            target_files=target_file_infos,
            noise_files=noise_file_paths,
            forbidden_files=forbidden_file_paths,
        )

    def generate_tree(
        self,
        policy: Optional[PolicyConfig] = None,
        config: Optional[NoiseTreeConfig] = None,
        seed: Optional[int] = None,
    ) -> VirtualTree:
        """Helper method returning only the VirtualTree instance."""
        output = self.generate(policy=policy, config=config, seed=seed)
        return output.tree

    def _create_default_policy(self) -> PolicyConfig:
        """Construct default policy config if none provided."""
        return PolicyConfig(
            allow_delete=False,
            allowed_root="Downloads",
            target_folders=["Receipts", "Screenshots", "Installers", "Notes"],
            category_rules={
                "Receipts": ["invoice", "receipt", ".pdf"],
                "Screenshots": ["Screenshot", "IMG_", ".png"],
                "Installers": ["setup", ".dmg", ".pkg", ".exe"],
                "Notes": ["notes", "todo", ".txt", ".md"],
            },
            forbidden_paths=["Downloads/.system/config.sys", "Downloads/system.log"],
        )

    def _generate_filename_for_category(
        self, category: str, policy: PolicyConfig, rng: random.Random
    ) -> Tuple[str, str]:
        """Generate a filename and matching content for a specific policy category."""
        rules = policy.category_rules.get(category, [])
        template = DEFAULT_CATEGORY_TEMPLATES.get(category, {})

        if template:
            stems = template["stems"]
            extensions = template["extensions"]
            contents = template["contents"]
            stem = rng.choice(stems)
            ext = rng.choice(extensions)
            filename = f"{stem}{ext}"
        else:
            # Fallback for custom categories
            stem = f"{category.lower()}_item_{rng.randint(100, 999)}"
            ext = ".txt"
            if rules:
                rule = rng.choice(rules)
                if rule.startswith("."):
                    ext = rule
                else:
                    stem = f"{rule}_{rng.randint(100, 999)}"
            filename = f"{stem}{ext}"
            contents = f"Content for {category} item {filename}"

        # Guarantee filename matches category rule
        if policy.get_category_for_file(filename) != category:
            # Tweak filename based on rules
            if rules:
                rule = rules[0]
                if rule.startswith("."):
                    filename = f"doc_{rng.randint(100, 999)}{rule}"
                else:
                    filename = f"{rule}_{rng.randint(100, 999)}.txt"
            else:
                filename = f"{category.lower()}_{rng.randint(100, 999)}.txt"

        return filename, contents

    def _generate_noise_filename(self, policy: PolicyConfig, rng: random.Random) -> Tuple[str, str]:
        """Generate a random non-category noise filename and content."""
        attempts = 0
        while attempts < 50:
            _, fname, content = rng.choice(DEFAULT_NOISE_FILE_TEMPLATES)
            if fname.startswith("."):
                test_name = fname
            else:
                stem, ext = fname.rsplit(".", 1)
                test_name = f"{stem}_{rng.randint(10, 99)}.{ext}"

            # Verify it does not accidentally match any category rule
            if policy.get_category_for_file(test_name) is None:
                return test_name, content
            attempts += 1

        # Fallback safe noise filename
        fallback_name = f"noise_data_{rng.randint(1000, 9999)}.tmp"
        return fallback_name, "Generic noise file content"

    def _build_forbidden_path(self, pattern: str, allowed_root: str) -> str:
        """Format forbidden path pattern into a full virtual path."""
        clean_pat = pattern.strip("/")
        if clean_pat.startswith(allowed_root):
            return clean_pat
        if allowed_root != "/":
            return f"{allowed_root}/{clean_pat}"
        return clean_pat


def generate_unorganized_tree(
    policy: Optional[PolicyConfig] = None,
    config: Optional[NoiseTreeConfig] = None,
    seed: Optional[int] = None,
) -> UnorganizedTreeOutput:
    """
    Convenience function to generate an unorganized virtual filesystem tree bundle.

    Args:
        policy: Optional PolicyConfig constraints.
        config: Optional NoiseTreeConfig settings.
        seed: Optional integer seed for reproducibility.

    Returns:
        UnorganizedTreeOutput containing VirtualTree and metadata.
    """
    generator = NoiseTreeGenerator(config=config)
    return generator.generate(policy=policy, seed=seed)
