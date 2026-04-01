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
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import qasync
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)

from gui.left_panel import LeftPanel
from gui.middle_panel import MiddlePanel
from bezier.bezier import CubicBezierSpline

from cflib2 import Crazyflie, LinkContext
from cflib2.memory import CompressedSegment, CompressedStart
from cflib2.toc_cache import FileTocCache


TRAJECTORY_IDS = {
    "takeoff": 1,
    "show_forward": 2,
    "show_backward": 3,
    "landing": 4,
}

TAKEOFF_HEIGHT = 1.0
TAKEOFF_DURATION = 2.0
GO_TO_START_DURATION = 2.0
GO_TO_PAD_DURATION = 3.0
LOG_INTERVAL = 100  # ms
STITCH_POINT_REPEAT = 10


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


async def drain_log(log_stream: object, last_log: dict[str, Any]) -> None:
    """Continuously drain a log stream, keeping only the most recent reading."""
    while True:
        data = await log_stream.next()
        last_log["data"] = data.data


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
        self._connect_task: asyncio.Task | None = None
        self._fly_task: asyncio.Task | None = None
        self._connected_cfs: list[object] = []
        self._link_context: object | None = None
        self._live_log_tasks: list[asyncio.Task] = []
        self._live_log_streams: list[object] = []
        self._latest_positions: list[dict[str, Any]] = []
        self._live_view_task: asyncio.Task | None = None

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
        self.panel_left.connect_requested.connect(self._on_connect_clicked)
        self.panel_left.disconnect_requested.connect(self._on_disconnect_clicked)
        self.panel_left.fly_requested.connect(self._on_fly_clicked)
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

    def _on_connect_clicked(self, base_address: str, num_drones: int) -> None:
        if self._connect_task is not None and not self._connect_task.done():
            return

        self._connect_task = asyncio.create_task(
            self._connect_drones(base_address, num_drones)
        )

    def _on_disconnect_clicked(self) -> None:
        asyncio.create_task(self._disconnect_and_set_idle())

    def _on_fly_clicked(self) -> None:
        if self._fly_task is not None and not self._fly_task.done():
            return
        self._fly_task = asyncio.create_task(self._fly_connected_drones())

    async def _connect_drones(self, base_address: str, num_drones: int) -> None:
        await self._sm.transition(AppState.CONNECTING)

        uris = [f"{base_address}{index:02X}" for index in range(1, num_drones + 1)]

        if self._connected_cfs:
            await self._disconnect_all_drones()

        self._link_context = LinkContext()

        try:
            self._connected_cfs = list(
                await asyncio.gather(
                    *[
                        Crazyflie.connect_from_uri(
                            self._link_context,
                            uri,
                            FileTocCache("cache"),
                        )
                        for uri in uris
                    ]
                )
            )
        except Exception as exc:
            await self._disconnect_all_drones()
            await self._sm.transition(AppState.ERROR)
            self.panel_left.set_fly_enabled(False)
            QMessageBox.critical(self, "Connection Failed", f"Could not connect to all drones: {exc}")
            return

        print(f"Connected to {len(self._connected_cfs)} drone(s): {', '.join(uris)}")
        self.panel_left.set_fly_enabled(True)
        await self._sm.transition(AppState.READY)

    async def _disconnect_all_drones(self) -> None:
        if not self._connected_cfs:
            return

        disconnect_tasks = [cf.disconnect() for cf in self._connected_cfs]
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        self._connected_cfs = []
        self._link_context = None
        self.panel_left.set_fly_enabled(False)

    async def _disconnect_and_set_idle(self) -> None:
        await self._disconnect_all_drones()
        await self._sm.transition(AppState.IDLE)

    @staticmethod
    def _load_phase_waypoints(
        csv_path: Path,
        n_drones: int,
        prepend_points: list[np.ndarray] | None = None,
        reverse: bool = False,
    ) -> list[np.ndarray]:
        df = pd.read_csv(csv_path)
        waypoints_per_drone: list[np.ndarray] = []

        if prepend_points is not None and len(prepend_points) != n_drones:
            raise ValueError(
                f"{csv_path.name} prepend point count ({len(prepend_points)}) does not match drone count ({n_drones})"
            )

        for idx in range(n_drones):
            try:
                xs = df[f"x{idx+20}"].values
                ys = df[f"y{idx+20}"].values
                zs = df[f"z{idx+20}"].values
            except KeyError as exc:
                raise ValueError(
                    f"{csv_path.name} is missing columns for drone index {idx} (x{idx}, y{idx}, z{idx})"
                ) from exc

            waypoints = np.column_stack((xs, ys, zs))
            if reverse:
                waypoints = waypoints[::-1]

            if prepend_points is not None:
                repeated_prepend = np.repeat(
                    prepend_points[idx][np.newaxis, :],
                    STITCH_POINT_REPEAT,
                    axis=0,
                )
                waypoints = np.vstack((repeated_prepend, waypoints))

            repeated_point = np.repeat(
                waypoints[-1][np.newaxis, :],
                STITCH_POINT_REPEAT,
                axis=0,
            )
            waypoints = np.vstack((waypoints, repeated_point))

            if len(waypoints) < 2:
                raise ValueError(f"{csv_path.name} must contain at least 2 waypoints per drone")
            waypoints_per_drone.append(waypoints)

        return waypoints_per_drone

    @classmethod
    def _load_phase_splines(
        cls,
        csv_path: Path,
        n_drones: int,
        prepend_points: list[np.ndarray] | None = None,
    ) -> list[CubicBezierSpline]:
        waypoints_per_drone = cls._load_phase_waypoints(
            csv_path,
            n_drones,
            prepend_points=prepend_points,
        )
        return [CubicBezierSpline.from_waypoints(waypoints) for waypoints in waypoints_per_drone]

    @classmethod
    def _load_phase_splines_reversed(cls, csv_path: Path, n_drones: int) -> list[CubicBezierSpline]:
        waypoints_per_drone = cls._load_phase_waypoints(csv_path, n_drones, reverse=True)
        return [CubicBezierSpline.from_waypoints(waypoints) for waypoints in waypoints_per_drone]

    @classmethod
    def _load_phase_last_points(cls, csv_path: Path, n_drones: int) -> list[np.ndarray]:
        waypoints_per_drone = cls._load_phase_waypoints(csv_path, n_drones)
        return [waypoints[-1] for waypoints in waypoints_per_drone]

    @staticmethod
    async def _upload_trajectory(
        cf: object,
        spline: CubicBezierSpline,
        trajectory_id: int,
        segment_duration: float,
        offset: int = 0,
    ) -> tuple[float, int]:
        coeffs = []
        for bezier in spline.beziers:
            coeffs.append(
                CompressedSegment(
                    duration=segment_duration,
                    x=[bezier.p1[0], bezier.p2[0], bezier.p3[0]],
                    y=[bezier.p1[1], bezier.p2[1], bezier.p3[1]],
                    z=[bezier.p1[2], bezier.p2[2], bezier.p3[2]],
                    yaw=[0.0, 0.0, 0.0],
                )
            )

        bytes_written = await cf.memory().write_compressed_trajectory(
            CompressedStart(spline.beziers[0].p0[0], spline.beziers[0].p0[1], spline.beziers[0].p0[2], 0.0),
            coeffs,
            start_addr=offset,
        ) 
        await cf.high_level_commander().define_trajectory(trajectory_id, offset, len(spline.beziers), 1)
        return len(spline.beziers) * segment_duration, bytes_written

    @staticmethod
    async def _read_pad_position(cf: object) -> tuple[float, float, float]:
        log = cf.log()
        block = await log.create_block()
        await block.add_variable("stateEstimate.x")
        await block.add_variable("stateEstimate.y")
        await block.add_variable("stateEstimate.z")
        log_stream = await block.start(LOG_INTERVAL)
        values = (await log_stream.next()).data
        return (
            float(values["stateEstimate.x"]),
            float(values["stateEstimate.y"]),
            float(values["stateEstimate.z"]),
        )

    async def _start_live_position_logging(self) -> None:
        self._latest_positions = [{"data": None} for _ in self._connected_cfs]
        self._live_log_streams = []
        self._live_log_tasks = []

        for idx, cf in enumerate(self._connected_cfs):
            log = cf.log()
            block = await log.create_block()
            await block.add_variable("stateEstimate.x")
            await block.add_variable("stateEstimate.y")
            await block.add_variable("stateEstimate.z")
            log_stream = await block.start(LOG_INTERVAL)
            self._live_log_streams.append(log_stream)
            self._live_log_tasks.append(
                asyncio.create_task(drain_log(log_stream, self._latest_positions[idx]))
            )

        if self._live_view_task is None or self._live_view_task.done():
            self._live_view_task = asyncio.create_task(self._run_live_view_updates())

    async def _run_live_view_updates(self) -> None:
        while True:
            positions: list[tuple[float, float, float] | None] = []
            for last in self._latest_positions:
                data = last.get("data")
                if not data:
                    positions.append(None)
                    continue
                positions.append(
                    (
                        float(data["stateEstimate.x"]),
                        float(data["stateEstimate.y"]),
                        float(data["stateEstimate.z"]),
                    )
                )
            self.panel_center.update_live_positions(positions)
            await asyncio.sleep(0.05)

    async def _stop_live_position_logging(self) -> None:
        for task in self._live_log_tasks:
            task.cancel()
        if self._live_log_tasks:
            await asyncio.gather(*self._live_log_tasks, return_exceptions=True)
        self._live_log_tasks = []

        if self._live_view_task is not None:
            self._live_view_task.cancel()
            await asyncio.gather(self._live_view_task, return_exceptions=True)
            self._live_view_task = None

        for stream in self._live_log_streams:
            stop = getattr(stream, "stop", None)
            if callable(stop):
                result = stop()
                if asyncio.iscoroutine(result):
                    await result
        self._live_log_streams = []
        self._latest_positions = []

    async def _fly_connected_drones(self) -> None:
        if not self._connected_cfs:
            QMessageBox.warning(self, "Not Connected", "Connect to drones before starting flight.")
            return

        phase_paths = self.panel_left.get_phase_csv_paths()
        if phase_paths is None:
            QMessageBox.warning(self, "Missing CSV", "Select a valid phase CSV folder first.")
            return

        takeoff_csv_path, active_csv_path, landing_csv_path = phase_paths
        n_drones = len(self._connected_cfs)
        num_trials = max(1, self.panel_left.get_num_trials())

        dt_start = self.panel_left.get_dt_start_seconds()
        dt_show = self.panel_left.get_dt_show_seconds()
        phase_segment_durations = {
            "takeoff": dt_start,
            "show_forward": dt_show,
            "show_backward": dt_show,
            "landing": dt_start,
        }

        try:
            takeoff_splines = self._load_phase_splines(takeoff_csv_path, n_drones)
            takeoff_last_points = self._load_phase_last_points(takeoff_csv_path, n_drones)

            show_forward_splines = self._load_phase_splines(
                active_csv_path,
                n_drones,
                prepend_points=takeoff_last_points,
            )
            show_backward_splines = self._load_phase_splines_reversed(active_csv_path, n_drones)

            show_last_points = self._load_phase_last_points(active_csv_path, n_drones)
            landing_splines = self._load_phase_splines(
                landing_csv_path,
                n_drones,
                prepend_points=show_last_points,
            )
        except Exception as exc:
            await self._sm.transition(AppState.ERROR)
            QMessageBox.critical(self, "Invalid Trajectory Data", str(exc))
            return

        await self._sm.transition(AppState.FLYING)
        self.panel_left.set_fly_enabled(False)
        self.panel_center.start_live_mode(n_drones)

        try:
            print("Applying initial controller parameters...")
            for cf in self._connected_cfs:
                param = cf.param()
                param.set("landingCrtl.hOffset", 0.02)
                param.set("landingCrtl.hDuration", 1.0)
                param.set("ctrlMel.ki_z", 1.5)
                param.set("stabilizer.controller", 1)

            print("Reading pad positions...")
            pad_positions = list(
                await asyncio.gather(
                    *[self._read_pad_position(cf) for cf in self._connected_cfs]
                )
            )

            print("Starting live position logger streams...")
            await self._start_live_position_logging()

            # Upload all trajectories for each drone before arming.
            print("Uploading takeoff/show-forward/show-backward/landing trajectories...")
            phase_splines = {
                "takeoff": takeoff_splines,
                "show_forward": show_forward_splines,
                "show_backward": show_backward_splines,
                "landing": landing_splines,
            }
            phase_total_durations: dict[str, float] = {}
            offsets = [0 for _ in self._connected_cfs]

            for phase_name in ("takeoff", "show_forward", "show_backward", "landing"):
                phase_id = TRAJECTORY_IDS[phase_name]
                seg_duration = phase_segment_durations[phase_name]
                upload_results = await asyncio.gather(
                    *[
                        self._upload_trajectory(
                            cf,
                            phase_splines[phase_name][idx],
                            phase_id,
                            seg_duration,
                            offset=offsets[idx],
                        )
                        for idx, cf in enumerate(self._connected_cfs)
                    ]
                )
                phase_total_durations[phase_name] = (
                    max(duration for duration, _ in upload_results) if upload_results else 0.0
                )
                for idx, (_, bytes_written) in enumerate(upload_results):
                    offsets[idx] += bytes_written

            print("Arming drones...")
            await asyncio.gather(
                *[cf.platform().send_arming_request(True) for cf in self._connected_cfs]
            )
            await asyncio.sleep(1.0)

            print("Taking off...")
            await asyncio.gather(
                *[
                    cf.high_level_commander().take_off(
                        TAKEOFF_HEIGHT,
                        None,
                        TAKEOFF_DURATION,
                        None,
                    )
                    for cf in self._connected_cfs
                ]
            )
            await asyncio.sleep(TAKEOFF_DURATION + 1.0)

            print("Moving to takeoff trajectory start position...")
            await asyncio.gather(
                *[
                    cf.high_level_commander().go_to(
                        takeoff_splines[idx].beziers[0].p0[0],
                        takeoff_splines[idx].beziers[0].p0[1],
                        takeoff_splines[idx].beziers[0].p0[2],
                        0.0,
                        GO_TO_START_DURATION,
                        relative=False,
                        linear=False,
                        group_mask=None,
                    )
                    for idx, cf in enumerate(self._connected_cfs)
                ]
            )
            await asyncio.sleep(GO_TO_START_DURATION + 1.0)

            # Run each uploaded trajectory in order.
            print("Starting takeoff trajectory...")
            self.panel_center.set_live_phase_curves(
                "takeoff",
                takeoff_splines,
                phase_segment_durations["takeoff"],
            )
            await asyncio.gather(
                *[
                    cf.high_level_commander().start_trajectory(
                        TRAJECTORY_IDS["takeoff"],
                        1.0,
                        False,
                        False,
                        False,
                        None,
                    )
                    for cf in self._connected_cfs
                ]
            )
            await asyncio.sleep(phase_total_durations["takeoff"] + 0.5)

            print(f"Starting show trajectories for {num_trials} trial(s)...")
            print("Show pass 1: forward")
            self.panel_center.set_live_phase_curves(
                "show_forward",
                show_forward_splines,
                phase_segment_durations["show_forward"],
            )
            await asyncio.gather(
                *[
                    cf.high_level_commander().start_trajectory(
                        TRAJECTORY_IDS["show_forward"],
                        1.0,
                        False,
                        False,
                        False,
                        None,
                    )
                    for cf in self._connected_cfs
                ]
            )
            await asyncio.sleep(phase_total_durations["show_forward"] + 0.5)

            # Match simulation behavior: F, (B, F) repeated num_trials-1 times.
            for trial_idx in range(2, num_trials + 1):
                print(f"Show pass {trial_idx}: backward")
                self.panel_center.set_live_phase_curves(
                    "show_backward",
                    show_backward_splines,
                    phase_segment_durations["show_backward"],
                )
                await asyncio.gather(
                    *[
                        cf.high_level_commander().start_trajectory(
                            TRAJECTORY_IDS["show_backward"],
                            1.0,
                            False,
                            False,
                            False,
                            None,
                        )
                        for cf in self._connected_cfs
                    ]
                )
                await asyncio.sleep(phase_total_durations["show_backward"] + 0.5)

                print(f"Show pass {trial_idx}: forward")
                self.panel_center.set_live_phase_curves(
                    "show_forward",
                    show_forward_splines,
                    phase_segment_durations["show_forward"],
                )
                await asyncio.gather(
                    *[
                        cf.high_level_commander().start_trajectory(
                            TRAJECTORY_IDS["show_forward"],
                            1.0,
                            False,
                            False,
                            False,
                            None,
                        )
                        for cf in self._connected_cfs
                    ]
                )
                await asyncio.sleep(phase_total_durations["show_forward"] + 0.5)

            print("Starting landing trajectory...")
            self.panel_center.set_live_phase_curves(
                "landing",
                landing_splines,
                phase_segment_durations["landing"],
            )
            await asyncio.gather(
                *[
                    cf.high_level_commander().start_trajectory(
                        TRAJECTORY_IDS["landing"],
                        1.0,
                        False,
                        False,
                        False,
                        None,
                    )
                    for cf in self._connected_cfs
                ]
            )
            await asyncio.sleep(phase_total_durations["landing"] + 0.5)

            print("Returning to pad positions...")
            await asyncio.gather(
                *[
                    cf.high_level_commander().go_to(
                        pad_positions[idx][0],
                        pad_positions[idx][1],
                        pad_positions[idx][2] + TAKEOFF_HEIGHT,
                        0.0,
                        GO_TO_PAD_DURATION,
                        relative=False,
                        linear=False,
                        group_mask=None,
                    )
                    for idx, cf in enumerate(self._connected_cfs)
                ]
            )
            await asyncio.sleep(GO_TO_PAD_DURATION + 0.5)

            print("Applying pre-landing controller parameters...")
            for cf in self._connected_cfs:
                param = cf.param()
                param.set("landingCrtl.hOffset", 0.05)
                param.set("landingCrtl.hDuration", 1.0)
                param.set("ctrlMel.ki_z", 1.5)
                param.set("stabilizer.controller", 1)
            await asyncio.sleep(2.5)

            print("Landing...")
            await self._sm.transition(AppState.LANDING)
            await asyncio.gather(
                *[
                    cf.high_level_commander().land(pad_positions[idx][2], None, 2.0, None)
                    for idx, cf in enumerate(self._connected_cfs)
                ]
            )
            await asyncio.sleep(2.5)

            await asyncio.gather(
                *[cf.high_level_commander().stop(None) for cf in self._connected_cfs]
            )
            await asyncio.gather(
                *[cf.platform().send_arming_request(False) for cf in self._connected_cfs]
            )

            await self._sm.transition(AppState.READY)
            QMessageBox.information(self, "Flight Complete", "All trajectories executed successfully.")
        except Exception as exc:
            await self._sm.transition(AppState.ERROR)
            QMessageBox.critical(self, "Flight Failed", f"Flight sequence failed: {exc}")
        finally:
            await self._stop_live_position_logging()
            self.panel_center.stop_live_mode()
            if self._connected_cfs:
                self.panel_left.set_fly_enabled(True)

    def _on_state_changed(self, state: AppState) -> None:
        """React to every state transition. Expand per-state logic here."""
        self._update_status_bar(state)

        if state == AppState.IDLE:
            pass  # TODO: reset UI
        elif state == AppState.CONNECTING:
            pass  # TODO: show spinner / disable controls
        elif state == AppState.READY:
            pass  # TODO: enable fly button
        elif state == AppState.SIMULATING:
            pass  # TODO: run simulation flow
        elif state == AppState.FLYING:
            pass  # TODO: start live visualisation
        elif state == AppState.LANDING:
            pass  # TODO: show landing progress
        elif state == AppState.ERROR:
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
