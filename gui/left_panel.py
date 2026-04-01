#!/usr/bin/env python3
"""Left panel widget for CSV folder selection and validation."""

import csv
from enum import Enum, auto
from pathlib import Path
import re
import yaml

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class LeftPanel(QFrame):
    simulate_requested = pyqtSignal()

    class PanelState(Enum):
        LOAD_CSV = auto()
        ERROR = auto()
        CONFIGURING_SHOW = auto()

    VALID_STYLE = "color: #1f7a1f;"
    INVALID_STYLE = "color: #b00020;"
    NEUTRAL_STYLE = "color: #666666;"
    REQUIRED_CSV_FILES = {
        "takeoff_phase.csv",
        "active_phase.csv",
        "landing_phase.csv",
    }
    CONFIG_FILE_NAME = "mad_config.yaml"

    def __init__(self) -> None:
        super().__init__()
        self._state = self.PanelState.LOAD_CSV
        self._last_validation_message = "No folder selected"

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Fleet")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(6)

        self._folder_input = QLineEdit()
        self._folder_input.setPlaceholderText("csv folder...")
        self._folder_input.textChanged.connect(self._on_folder_changed)
        row.addWidget(self._folder_input, stretch=1)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_folder)
        row.addWidget(browse_btn)

        layout.addLayout(row)

        self._validation_output = QLineEdit()
        self._validation_output.setReadOnly(True)
        layout.addWidget(self._validation_output)

        self._dt_start_input = QLineEdit()
        self._dt_show_input = QLineEdit()
        self._num_trials_input = QLineEdit()
        self._last_valid_dt_start = "0.1"
        self._last_valid_dt_show = "0.1"
        self._last_valid_num_trials = "1"
        self._dt_start_input.setText(self._last_valid_dt_start)
        self._dt_show_input.setText(self._last_valid_dt_show)
        self._num_trials_input.setText(self._last_valid_num_trials)

        for number_input in (
            self._dt_start_input,
            self._dt_show_input,
            self._num_trials_input,
        ):
            number_input.setPlaceholderText("0")

        self._dt_start_input.returnPressed.connect(
            lambda: self._validate_and_commit_float(
                self._dt_start_input,
                "_last_valid_dt_start",
            )
        )
        self._dt_show_input.returnPressed.connect(
            lambda: self._validate_and_commit_float(
                self._dt_show_input,
                "_last_valid_dt_show",
            )
        )
        self._num_trials_input.returnPressed.connect(
            lambda: self._validate_and_commit_int(
                self._num_trials_input,
                "_last_valid_num_trials",
            )
        )

        config_layout = QFormLayout()
        config_layout.setSpacing(8)
        config_layout.addRow("Dt start", self._dt_start_input)
        config_layout.addRow("Dt show", self._dt_show_input)
        config_layout.addRow("Number Trials", self._num_trials_input)
        layout.addLayout(config_layout)

        self._save_config_btn = QPushButton("Save Config")
        self._save_config_btn.clicked.connect(self._save_config)
        layout.addWidget(self._save_config_btn)

        self._simulate_btn = QPushButton("Simulate")
        self._simulate_btn.clicked.connect(self.simulate_requested.emit)
        layout.addWidget(self._simulate_btn)

        self._apply_state_ui()

        layout.addStretch(1)

    def _browse_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if selected:
            self._folder_input.setText(selected)

    def _on_folder_changed(self, folder: str) -> None:
        valid, message = self._validate_folder(folder)
        self._last_validation_message = message

        if not folder.strip():
            self._transition_to(self.PanelState.LOAD_CSV)
        elif valid:
            self._transition_to(self.PanelState.CONFIGURING_SHOW)
        else:
            self._transition_to(self.PanelState.ERROR)

        if folder.strip() and not valid:
            print(f"[Validation] {message}")

        if folder.strip() and valid:
            self._load_config_if_exists(Path(folder.strip()))

    def _transition_to(self, state: PanelState) -> None:
        self._state = state
        self._apply_state_ui()

    def _set_configuration_fields_enabled(self, enabled: bool) -> None:
        for number_input in (
            self._dt_start_input,
            self._dt_show_input,
            self._num_trials_input,
        ):
            number_input.setEnabled(enabled)

    def _apply_state_ui(self) -> None:
        if self._state == self.PanelState.LOAD_CSV:
            self._validation_output.setText("Load CSV folder")
            self._validation_output.setStyleSheet(self.NEUTRAL_STYLE)
            self._set_configuration_fields_enabled(False)
        elif self._state == self.PanelState.CONFIGURING_SHOW:
            self._validation_output.setText("Successfully Loaded Show")
            self._validation_output.setStyleSheet(self.VALID_STYLE)
            self._set_configuration_fields_enabled(True)
        else:
            self._validation_output.setText(f"Invalid Format")
            self._validation_output.setStyleSheet(self.INVALID_STYLE)
            self._set_configuration_fields_enabled(False)

    def _restore_last_valid_value(self, input_widget: QLineEdit, last_valid_attr: str) -> None:
        input_widget.setText(getattr(self, last_valid_attr))

    def _validate_and_commit_float(self, input_widget: QLineEdit, last_valid_attr: str) -> None:
        text = input_widget.text().strip()

        try:
            value = float(text)
        except ValueError:
            self._restore_last_valid_value(input_widget, last_valid_attr)
            return

        if value <= 0:
            self._restore_last_valid_value(input_widget, last_valid_attr)
            return

        setattr(self, last_valid_attr, text)
        input_widget.setText(text)

    def _validate_and_commit_int(self, input_widget: QLineEdit, last_valid_attr: str) -> None:
        text = input_widget.text().strip()

        if not text.isdigit():
            self._restore_last_valid_value(input_widget, last_valid_attr)
            return

        value = int(text)
        if value <= 0:
            self._restore_last_valid_value(input_widget, last_valid_attr)
            return

        setattr(self, last_valid_attr, text)
        input_widget.setText(text)

    def _get_config_path(self) -> Path | None:
        folder = self._folder_input.text().strip()
        if not folder:
            return None
        return Path(folder) / self.CONFIG_FILE_NAME

    def _build_config_data(self) -> dict[str, float | int]:
        return {
            "dt_start": self.get_dt_start_seconds(),
            "dt_show": self.get_dt_show_seconds(),
            "num_trials": self.get_num_trials(),
        }

    def _set_config_data(self, data: dict) -> None:
        dt_start = data.get("dt_start")
        dt_show = data.get("dt_show")
        num_trials = data.get("num_trials")

        if isinstance(dt_start, (int, float)) and float(dt_start) > 0:
            value = str(float(dt_start))
            self._last_valid_dt_start = value
            self._dt_start_input.setText(value)

        if isinstance(dt_show, (int, float)) and float(dt_show) > 0:
            value = str(float(dt_show))
            self._last_valid_dt_show = value
            self._dt_show_input.setText(value)

        if isinstance(num_trials, int) and num_trials > 0:
            value = str(num_trials)
            self._last_valid_num_trials = value
            self._num_trials_input.setText(value)

    def _load_config_if_exists(self, folder_path: Path) -> None:
        config_path = folder_path / self.CONFIG_FILE_NAME
        if not config_path.is_file():
            return

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError):
            return

        if isinstance(loaded, dict):
            self._set_config_data(loaded)

    def _save_config(self) -> None:
        config_path = self._get_config_path()
        if config_path is None:
            QMessageBox.warning(self, "No Folder", "Select a valid CSV folder first.")
            return

        folder_path = config_path.parent
        valid, message = self._validate_folder(str(folder_path))
        if not valid:
            QMessageBox.warning(self, "Invalid Folder", message)
            return

        if config_path.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite Config",
                f"{self.CONFIG_FILE_NAME} already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        payload = self._build_config_data()

        try:
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)
        except OSError as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not save config: {exc}")
            return

        QMessageBox.information(
            self,
            "Config Saved",
            f"Configuration saved to {config_path.name}.",
        )

    @staticmethod
    def _csv_header_has_xyz_groups(columns: list[str]) -> bool:
        pattern = re.compile(r"^([xyz])(\d+)$")
        groups: dict[str, set[int]] = {"x": set(), "y": set(), "z": set()}

        for name in columns:
            match = pattern.match(name.strip())
            if match:
                axis, index = match.groups()
                groups[axis].add(int(index))

        return bool(groups["x"] & groups["y"] & groups["z"])

    def _validate_folder(self, folder: str) -> tuple[bool, str]:
        folder_text = folder.strip()
        if not folder_text:
            return False, "No folder selected"

        path = Path(folder_text)
        if not path.exists() or not path.is_dir():
            return False, "Invalid folder path"

        missing_files = [
            name for name in sorted(self.REQUIRED_CSV_FILES)
            if not (path / name).is_file()
        ]
        if missing_files:
            return False, f"Missing required CSV file(s): {', '.join(missing_files)}"

        invalid_files: list[str] = []
        for name in sorted(self.REQUIRED_CSV_FILES):
            csv_path = path / name
            try:
                with csv_path.open("r", newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, [])
                if not self._csv_header_has_xyz_groups(header):
                    invalid_files.append(name)
            except OSError:
                invalid_files.append(name)

        if invalid_files:
            return False, (
                "Invalid data format in: "
                + ", ".join(invalid_files)
                + ". Expected xN,yN,zN columns"
            )

        return True, "Valid data in required phase CSV files"

    def get_active_phase_csv_path(self) -> Path | None:
        folder = self._folder_input.text().strip()
        if not folder:
            return None

        path = Path(folder)
        active_csv = path / "active_phase.csv"
        if not active_csv.is_file():
            return None
        return active_csv

    def get_phase_csv_paths(self) -> tuple[Path, Path, Path] | None:
        folder = self._folder_input.text().strip()
        if not folder:
            return None

        path = Path(folder)
        takeoff_csv = path / "takeoff_phase.csv"
        active_csv = path / "active_phase.csv"
        landing_csv = path / "landing_phase.csv"

        if not (takeoff_csv.is_file() and active_csv.is_file() and landing_csv.is_file()):
            return None

        return takeoff_csv, active_csv, landing_csv

    def get_dt_start_seconds(self) -> float:
        text = self._dt_start_input.text().strip()
        if not text:
            text = self._last_valid_dt_start or self._last_valid_dt_show

        fallback = self._last_valid_dt_start or self._last_valid_dt_show
        try:
            value = float(text)
        except ValueError:
            value = float(fallback)

        if value <= 0:
            value = float(fallback)

        self._last_valid_dt_start = str(value)
        self._dt_start_input.setText(self._last_valid_dt_start)
        return value

    def get_dt_show_seconds(self) -> float:
        text = self._dt_show_input.text().strip()
        if not text:
            text = self._last_valid_dt_show

        try:
            value = float(text)
        except ValueError:
            value = float(self._last_valid_dt_show)

        if value <= 0:
            value = float(self._last_valid_dt_show)

        self._last_valid_dt_show = str(value)
        self._dt_show_input.setText(self._last_valid_dt_show)
        return value

    def get_num_trials(self) -> int:
        text = self._num_trials_input.text().strip()
        if not text:
            text = self._last_valid_num_trials or "1"

        fallback = self._last_valid_num_trials or "1"
        if not text.isdigit() or int(text) <= 0:
            value = int(fallback)
        else:
            value = int(text)

        self._last_valid_num_trials = str(value)
        self._num_trials_input.setText(self._last_valid_num_trials)
        return value
