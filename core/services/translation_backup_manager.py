import os
import time
from pathlib import Path
from typing import List, Optional


class TranslationBackupManager:
    def __init__(self, backup_dir: Optional[Path] = None):
        if backup_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.backup_dir = project_root / "Translation_backup"
        else:
            self.backup_dir = Path(backup_dir)
        self.current_batch_backup_path: Optional[Path] = None
        self.current_batch_files_signature: Optional[str] = None

    def ensure_dir(self):
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _compute_files_signature(self, input_files: List[str]) -> str:
        if not input_files:
            return ""
        return ",".join(sorted(Path(f).name for f in input_files))

    def save_single_raw(self, image_path: str, raw_content: str) -> Path:
        self.ensure_dir()
        base_name = Path(image_path).stem
        target_path = self.backup_dir / f"{base_name}_translation.txt"
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(raw_content.strip())
        return target_path

    def save_single_rendered(self, image_path: str, rendered_content: str) -> Path:
        self.ensure_dir()
        base_name = Path(image_path).stem
        target_path = self.backup_dir / f"{base_name}_translation.txt"
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(rendered_content.strip())
        return target_path

    def _generate_new_batch_path(self) -> Path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        target_path = self.backup_dir / f"batch_translation_{timestamp}.txt"
        counter = 1
        while target_path.exists():
            target_path = self.backup_dir / f"batch_translation_{timestamp}_{counter}.txt"
            counter += 1
        return target_path

    def save_batch_raw(
        self, raw_content: str, input_files: Optional[List[str]] = None
    ) -> Path:
        self.ensure_dir()
        target_path = self._generate_new_batch_path()
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(raw_content.strip())
        self.current_batch_backup_path = target_path
        if input_files:
            self.current_batch_files_signature = self._compute_files_signature(input_files)
        return target_path

    def save_batch_rendered(
        self, rendered_content: str, input_files: Optional[List[str]] = None
    ) -> Path:
        self.ensure_dir()
        sig = self._compute_files_signature(input_files) if input_files else None

        if (
            self.current_batch_backup_path is not None
            and self.current_batch_backup_path.exists()
        ):
            if (
                sig is None
                or not self.current_batch_files_signature
                or sig == self.current_batch_files_signature
            ):
                target_path = self.current_batch_backup_path
            else:
                target_path = self._generate_new_batch_path()
                self.current_batch_backup_path = target_path
                self.current_batch_files_signature = sig
        else:
            target_path = self._generate_new_batch_path()
            self.current_batch_backup_path = target_path
            self.current_batch_files_signature = sig

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(rendered_content.strip())
        return target_path

    def reset_batch_session(self):
        self.current_batch_backup_path = None
        self.current_batch_files_signature = None


_GLOBAL_BACKUP_MANAGER = TranslationBackupManager()


def get_backup_manager() -> TranslationBackupManager:
    return _GLOBAL_BACKUP_MANAGER
