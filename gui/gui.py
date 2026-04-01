#!/usr/bin/env python3
"""Swarm Controller GUI — skeleton.

Three-panel layout:
  Left   – drone fleet / connection status
  Center – 3-D visualisation
  Right  – controls & log

Entry point: run with  python gui.py
Requires:  PyQt6  qasync
"""

import asyncio
import sys
from enum import Enum, auto
from typing import Callable

import qasync
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QWidget,
)

from gui.left_panel import LeftPanel
from gui.middle_panel import MiddlePanel


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class AppState(Enum):
    IDLE       = auto()
    CONNECTING = auto()
    READY      = auto()
    SIMULATING = auto()
    FLYING     = auto()
    LANDING    = auto()
    ERROR      = auto()


_StateListener = Callable[[AppState], None]


class StateMachine:
    """Async state machine that drives the swarm controller lifecycle."""

    def __init__(self) -> None:
        self._state: AppState = AppState.IDLE
        self._listeners: list[_StateListener] = []

    # -- Public API ----------------------------------------------------------

    @property
    def state(self) -> AppState:
        return self._state

    def add_listener(self, cb: _StateListener) -> None:
        self._listeners.append(cb)

    async def transition(self, new_state: AppState) -> None:
        print(f"[SM] {self._state.name} -> {new_state.name}")
        self._state = new_state
        for cb in self._listeners:
            cb(new_state)

    # -- Main loop -----------------------------------------------------------

    async def run(self) -> None:
        """Drive state transitions.
        Expand with guard conditions, timers, and backend events later.
        """
        await self.transition(AppState.IDLE)

        # TODO: react to UI events and backend callbacks, e.g.:
        #   await self.transition(AppState.CONNECTING)
        #   await self._wait_for_all_connected()
        #   await self.transition(AppState.READY)
        #   ...


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, sm: StateMachine) -> None:
        super().__init__()
        self._sm = sm
        self._sm.add_listener(self._on_state_changed)
        self._transition_task: asyncio.Task | None = None

        self.setWindowTitle("Swarm Controller")
        self.resize(1600, 900)

        self._setup_ui()
        self._update_status_bar(sm.state)

    # -- UI setup ------------------------------------------------------------

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.panel_left   = LeftPanel()
        self.panel_left.simulate_requested.connect(self._on_simulate_clicked)
        self.panel_center = MiddlePanel()
        self.panel_right  = self._make_panel("Controls")   # buttons, log output

        root.addWidget(self.panel_left,   stretch=1)
        root.addWidget(self.panel_center, stretch=3)
        root.addWidget(self.panel_right,  stretch=1)

        # Status bar driven by the state machine
        self._status_label = QLabel()
        status_bar = QStatusBar()
        status_bar.addPermanentWidget(self._status_label)
        self.setStatusBar(status_bar)

    @staticmethod
    def _make_panel(title: str) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Raised)

        # Placeholder title — remove once real content is added
        label = QLabel(title, frame)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: gray; font-size: 18px;")
        label.setGeometry(0, 0, 200, 40)          # resized in resizeEvent later
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        return frame

    # -- State machine callback ---------------------------------------------

    def _on_simulate_clicked(self) -> None:
        phase_paths = self.panel_left.get_phase_csv_paths()
        if phase_paths is None:
            return

        takeoff_csv_path, active_csv_path, landing_csv_path = phase_paths
        dt_start = self.panel_left.get_dt_start_seconds()
        dt_show = self.panel_left.get_dt_show_seconds()
        num_trials = self.panel_left.get_num_trials()
        started = self.panel_center.start_simulation(
            takeoff_csv_path,
            active_csv_path,
            landing_csv_path,
            dt_start,
            dt_show,
            num_trials,
        )
        if not started:
            return

        # Schedule async transition without blocking the Qt event loop.
        self._transition_task = asyncio.create_task(
            self._sm.transition(AppState.SIMULATING)
        )

    def _on_state_changed(self, state: AppState) -> None:
        """React to every state transition. Expand per-state logic here."""
        self._update_status_bar(state)

        match state:
            case AppState.IDLE:
                pass  # TODO: reset UI
            case AppState.CONNECTING:
                pass  # TODO: show spinner / disable controls
            case AppState.READY:
                pass  # TODO: enable fly button
            case AppState.SIMULATING:
                pass  # TODO: run simulation flow
            case AppState.FLYING:
                pass  # TODO: start live visualisation
            case AppState.LANDING:
                pass  # TODO: show landing progress
            case AppState.ERROR:
                pass  # TODO: show error dialog

    def _update_status_bar(self, state: AppState) -> None:
        self._status_label.setText(f"State: {state.name}")


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------

async def _async_main() -> None:
    sm = StateMachine()
    window = MainWindow(sm)
    window.show()

    # Run the state machine as a concurrent background task
    asyncio.ensure_future(sm.run())


def main() -> None:
    app = QApplication(sys.argv)

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    with loop:
        loop.run_until_complete(_async_main())
        loop.run_forever()   # keeps running until the window is closed
