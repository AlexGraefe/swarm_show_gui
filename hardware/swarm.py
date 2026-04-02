"""Swarm hardware controller with an async state machine.

The :class:`Swarm` class owns all Crazyflie connection and flight logic.
It reports progress back to whatever object implements :class:`SwarmGUI`
(normally :class:`gui.gui.MainWindow`).
"""

import asyncio
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from bezier.bezier import CubicBezierSpline
from cflib2 import Crazyflie, LinkContext
from cflib2.memory import CompressedSegment, CompressedStart
from cflib2.toc_cache import FileTocCache


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAJECTORY_IDS = {
    "takeoff": 1,
    "show_forward": 2,
    "show_backward": 3,
    "landing": 4,
}

TAKEOFF_HEIGHT = 1.0
TAKEOFF_DURATION = 2.0
LANDING_DURATION = 3.0
GO_TO_START_DURATION = 2.0
GO_TO_PAD_DURATION = 3.0
GO_TO_END_DURATION = 1.0
LOG_INTERVAL = 100  # ms
STITCH_POINT_REPEAT = 10
STAGGER_STRIDE = 5   # launch/land every Nth drone per round (round 0: idx 0,4,8…; round 1: 1,5,9…)
STAGGER_DELAY  = TAKEOFF_DURATION + 0.5  # seconds between stagger groups

DRONE_IDX_MAPPING = {
    0: 18,
    1: 19,
    2: 20,
    3: 21,
    4: 22,
    5: 23,
    6: 24,
    7: 25,
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SwarmState(Enum):
    UNCONNECTED = auto()
    CONNECTED   = auto()
    FLYING      = auto()
    LANDED      = auto()
    ERROR       = auto()


# ---------------------------------------------------------------------------
# GUI callback protocol
# ---------------------------------------------------------------------------

class SwarmGUI(Protocol):
    """Callbacks the :class:`Swarm` uses to feed information back to the GUI.

    The GUI object passed to :class:`Swarm.__init__` must implement all of
    these methods (duck-typed via :class:`typing.Protocol`).
    """

    def on_swarm_state_changed(self, state: SwarmState) -> None:
        """Called whenever the swarm transitions to a new state."""
        ...

    def on_phase_changed(
        self,
        phase_name: str,
        splines: list,
        segment_duration: float,
    ) -> None:
        """Called when the active flight phase changes.

        *phase_name* is one of ``"takeoff"``, ``"show_forward"``,
        ``"show_backward"``, or ``"landing"``.  *splines* is the list of
        :class:`~bezier.bezier.CubicBezierSpline` for each drone.
        """
        ...

    def on_live_positions(
        self,
        positions: list,
    ) -> None:
        """Called with the latest drone positions during flight.

        *positions* is a ``list[tuple[float, float, float] | None]``, one
        element per drone.  ``None`` means no data has arrived yet for that
        drone.
        """
        ...

    def on_live_mode_started(self, n_drones: int) -> None:
        """Called just before the live-flight visualisation begins."""
        ...

    def on_live_mode_stopped(self) -> None:
        """Called after the live-flight visualisation has ended."""
        ...

    def on_fly_enabled(self, enabled: bool) -> None:
        """Tell the GUI whether the *Fly* button should be enabled."""
        ...

    def on_error(self, title: str, message: str) -> None:
        """Display an error to the user.  May be called from any async context."""
        ...


# ---------------------------------------------------------------------------
# Swarm
# ---------------------------------------------------------------------------

class Swarm:
    """Async state machine that controls a Crazyflie swarm.

    Usage::

        swarm = Swarm(gui)
        await swarm.connect("radio://0/80/2M/E7E7E7E7", 3)
        await swarm.fly(takeoff_csv, active_csv, landing_csv, dt_start, dt_show, trials)
        await swarm.emergency_land()
        await swarm.disconnect()
    """

    def __init__(self, gui: SwarmGUI) -> None:
        self._gui = gui
        self._state = SwarmState.UNCONNECTED
        self._connected_cfs: list[object] = []
        self._link_context: object | None = None
        self._live_log_streams: list[object] = []
        self._latest_positions: list[dict[str, Any]] = []
        self._connect_task: asyncio.Task | None = None
        self._fly_task: asyncio.Task | None = None

    # -- State ---------------------------------------------------------------

    @property
    def state(self) -> SwarmState:
        return self._state

    def _set_state(self, state: SwarmState) -> None:
        print(f"[Swarm] {self._state.name} -> {state.name}")
        self._state = state
        self._gui.on_swarm_state_changed(state)

    # -- Public API ----------------------------------------------------------

    def connect(self, base_address: str, num_drones: int) -> None:
        """Start connecting to *num_drones* drones derived from *base_address*.

        Returns immediately; progress is reported via :class:`SwarmGUI` callbacks.
        Ignored if a connection attempt is already in progress.
        """
        if self._connect_task is not None and not self._connect_task.done():
            return
        self._connect_task = asyncio.create_task(
            self._connect_impl(base_address, num_drones)
        )
        self._connect_task.add_done_callback(self._on_connect_task_done)

    async def _connect_impl(self, base_address: str, num_drones: int) -> None:
        """Internal coroutine that performs the actual connection sequence."""
        if self._connected_cfs:
            await self._disconnect_all()

        self._link_context = LinkContext()
        uris = [f"{base_address}{index:02X}" for index in range(1, num_drones + 1)]

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
            await self._disconnect_all()
            self._set_state(SwarmState.ERROR)
            self._gui.on_fly_enabled(False)
            self._gui.on_error("Connection Failed", f"Could not connect to all drones: {exc}")
            return

        print(f"Connected to {len(self._connected_cfs)} drone(s): {', '.join(uris)}")
        self._gui.on_fly_enabled(True)
        self._set_state(SwarmState.CONNECTED)

    async def disconnect(self) -> None:
        """Disconnect all drones.

        Cancels any in-progress fly sequence first.
        Transitions: * → UNCONNECTED.
        """
        if self._fly_task is not None and not self._fly_task.done():
            self._fly_task.cancel()
            await asyncio.gather(self._fly_task, return_exceptions=True)
            self._fly_task = None
        await self._disconnect_all()
        self._set_state(SwarmState.UNCONNECTED)

    async def emergency_land(self) -> None:
        """Immediately land all connected drones regardless of current state.

        Cancels any in-progress fly sequence first.
        Transitions: * → CONNECTED (so the operator can reconnect and retry).
        """
        if self._fly_task is not None and not self._fly_task.done():
            self._fly_task.cancel()
            await asyncio.gather(self._fly_task, return_exceptions=True)
            self._fly_task = None
        if not self._connected_cfs:
            return

        self._set_state(SwarmState.LANDED)
        print("Emergency land: commanding all drones to land...")
        try:
            await asyncio.gather(
                *[
                    cf.high_level_commander().land(0.0, None, 2.0, None)
                    for cf in self._connected_cfs
                ],
                return_exceptions=True,
            )
            await asyncio.sleep(3.0)
            await asyncio.gather(
                *[cf.high_level_commander().stop(None) for cf in self._connected_cfs],
                return_exceptions=True,
            )
            await asyncio.gather(
                *[cf.platform().send_arming_request(False) for cf in self._connected_cfs],
                return_exceptions=True,
            )
        except Exception as exc:
            print(f"Emergency land error: {exc}")
        finally:
            await self._stop_live_position_logging()
            self._gui.on_live_mode_stopped()
            self._set_state(SwarmState.CONNECTED)

    def fly(
        self,
        takeoff_csv: Path,
        active_csv: Path,
        landing_csv: Path,
        dt_start: float,
        dt_show: float,
        num_trials: int,
        wait_after_takeoff: float = 5.0,
        wait_between_passes: float = 5.0,
        wait_before_landing: float = 5.0,
    ) -> None:
        """Start the full flight sequence: takeoff → show → landing.

        Returns immediately; progress is reported via :class:`SwarmGUI` callbacks.
        Ignored if a flight is already in progress; shows an error if a stale
        task is present but the swarm is not actively flying.
        """
        if self._fly_task is not None:
            if self._fly_task.done():
                self._fly_task = None
            elif self._state in (SwarmState.FLYING, SwarmState.LANDED):
                self._gui.on_error(
                    "Flight In Progress", "A flight sequence is already running."
                )
                return
            else:
                # Stale task not matching current state — cancel and proceed.
                self._fly_task.cancel()
                self._fly_task = None
        self._fly_task = asyncio.create_task(
            self._fly_impl(takeoff_csv, active_csv, landing_csv, dt_start, dt_show, num_trials,
                           wait_after_takeoff, wait_between_passes, wait_before_landing)
        )
        self._fly_task.add_done_callback(self._on_fly_task_done)

    async def _fly_impl(
        self,
        takeoff_csv: Path,
        active_csv: Path,
        landing_csv: Path,
        dt_start: float,
        dt_show: float,
        num_trials: int,
        wait_after_takeoff: float = 5.0,
        wait_between_passes: float = 5.0,
        wait_before_landing: float = 5.0,
    ) -> None:
        """Internal coroutine that performs the full flight sequence."""
        if not self._connected_cfs:
            self._gui.on_error("Not Connected", "Connect to drones before starting flight.")
            return

        n_drones = len(self._connected_cfs)
        num_trials = max(1, num_trials)
        phase_segment_durations = {
            "takeoff": dt_start,
            "show_forward": dt_show,
            "show_backward": dt_show,
            "landing": dt_start,
        }

        try:
            takeoff_splines = self._load_phase_splines(takeoff_csv, n_drones)
            takeoff_last_points = self._load_phase_last_points(takeoff_csv, n_drones)

            show_forward_splines = self._load_phase_splines(
                active_csv,
                n_drones,
                prepend_points=takeoff_last_points,
            )
            show_backward_splines = self._load_phase_splines_reversed(active_csv, n_drones)

            show_last_points = self._load_phase_last_points(active_csv, n_drones)
            landing_splines = self._load_phase_splines(
                landing_csv,
                n_drones,
                prepend_points=show_last_points,
            )
        except Exception as exc:
            self._set_state(SwarmState.ERROR)
            self._gui.on_error("Invalid Trajectory Data", str(exc))
            return

        self._set_state(SwarmState.FLYING)
        self._gui.on_fly_enabled(False)
        self._gui.on_live_mode_started(n_drones)

        needs_landing = False
        try:
            print("Applying initial controller parameters...")
            for cf in self._connected_cfs:
                param = cf.param()
                param.set("colorLedBot.wrgb8888", 0x000000FF)
            #     param = cf.param()
            #     param.set("landingCrtl.hOffset", 0.02)
            #     param.set("landingCrtl.hDuration", 1.0)
            #     param.set("ctrlMel.ki_z", 1.5)
            #     param.set("stabilizer.controller", 1)

            print("Reading pad positions...")
            pad_positions = list(
                await asyncio.gather(
                    *[self._read_pad_position(cf) for cf in self._connected_cfs]
                )
            )

            misaligned = []
            for idx, pad_pos in enumerate(pad_positions):
                start = takeoff_splines[idx].beziers[0].p0
                dx = abs(pad_pos[0] - start[0])
                dy = abs(pad_pos[1] - start[1])
                if dx > 0.15 or dy > 0.15:
                    misaligned.append(
                        f"  Drone {idx + 1}: pad=({pad_pos[0]:.3f}, {pad_pos[1]:.3f}), "
                        f"start=({start[0]:.3f}, {start[1]:.3f}), "
                        f"dx={dx:.3f} m, dy={dy:.3f} m"
                    )
            if misaligned:
                raise ValueError(
                    "Pad position(s) too far from takeoff start (limit ±0.15 m):\n"
                    + "\n".join(misaligned)
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
            needs_landing = True
            await self._sleep_with_live_updates(1.0)

            print("Taking off...")
            stagger_groups = self._stagger_groups(n_drones)
            for group_idx, group in enumerate(stagger_groups):
                await asyncio.gather(
                    *[
                        self._connected_cfs[i].high_level_commander().take_off(
                            TAKEOFF_HEIGHT, None, TAKEOFF_DURATION, None
                        )
                        for i in group
                    ]
                )
                await self._sleep_with_live_updates(STAGGER_DELAY)
            # await self._sleep_with_live_updates(TAKEOFF_DURATION + 1.0)

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
            await self._sleep_with_live_updates(GO_TO_START_DURATION + 1.0)

            print("Starting takeoff trajectory...")
            self._gui.on_phase_changed(
                "takeoff", takeoff_splines, phase_segment_durations["takeoff"]
            )
            await asyncio.gather(
                *[
                    cf.high_level_commander().start_trajectory(
                        TRAJECTORY_IDS["takeoff"], 1.0, False, False, False, None
                    )
                    for cf in self._connected_cfs
                ]
            )
            await self._sleep_with_live_updates(phase_total_durations["takeoff"])

            if wait_after_takeoff > 0:
                print(f"Hovering {wait_after_takeoff}s after takeoff...")
                await self._sleep_with_live_updates(wait_after_takeoff)

            print(f"Starting show trajectories for {num_trials} trial(s)...")
            print("Show pass 1: forward")
            self._gui.on_phase_changed(
                "show_forward", show_forward_splines, phase_segment_durations["show_forward"]
            )
            await asyncio.gather(
                *[
                    cf.high_level_commander().start_trajectory(
                        TRAJECTORY_IDS["show_forward"], 1.0, False, False, False, None
                    )
                    for cf in self._connected_cfs
                ]
            )
            await self._sleep_with_live_updates(phase_total_durations["show_forward"])

            # Match simulation behaviour: F, (B, F) repeated num_trials-1 times.
            for trial_idx in range(2, num_trials + 1):
                if wait_between_passes > 0:
                    print(f"Waiting {wait_between_passes}s between passes...")
                    await self._sleep_with_live_updates(wait_between_passes)

                print(f"Show pass {trial_idx}: backward")
                self._gui.on_phase_changed(
                    "show_backward",
                    show_backward_splines,
                    phase_segment_durations["show_backward"],
                )
                await asyncio.gather(
                    *[
                        cf.high_level_commander().start_trajectory(
                            TRAJECTORY_IDS["show_backward"], 1.0, False, False, False, None
                        )
                        for cf in self._connected_cfs
                    ]
                )
                await self._sleep_with_live_updates(phase_total_durations["show_backward"])


                if wait_between_passes > 0:
                    print(f"Waiting {wait_between_passes}s between passes...")
                    await self._sleep_with_live_updates(wait_between_passes)

                print(f"Show pass {trial_idx}: forward")
                self._gui.on_phase_changed(
                    "show_forward", show_forward_splines, phase_segment_durations["show_forward"]
                )
                await asyncio.gather(
                    *[
                        cf.high_level_commander().start_trajectory(
                            TRAJECTORY_IDS["show_forward"], 1.0, False, False, False, None
                        )
                        for cf in self._connected_cfs
                    ]
                )
                await self._sleep_with_live_updates(phase_total_durations["show_forward"])


            if wait_before_landing > 0:
                print(f"Hovering {wait_before_landing}s before landing...")
                await self._sleep_with_live_updates(wait_before_landing)

            print("Starting landing trajectory...")
            self._gui.on_phase_changed(
                "landing", landing_splines, phase_segment_durations["landing"]
            )
            await asyncio.gather(
                *[
                    cf.high_level_commander().start_trajectory(
                        TRAJECTORY_IDS["landing"], 1.0, False, False, False, None
                    )
                    for cf in self._connected_cfs
                ]
            )
            await self._sleep_with_live_updates(phase_total_durations["landing"] + 0.5)

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
            await self._sleep_with_live_updates(GO_TO_PAD_DURATION + 0.5)

            print("Applying pre-landing controller parameters...")
            for cf in self._connected_cfs:
                param = cf.param()
                param.set("landingCrtl.hOffset", 0.03)
                param.set("landingCrtl.hDuration", 1.0)
                param.set("colorLedBot.wrgb8888", 0x00000000)
            #     param.set("ctrlMel.ki_z", 1.5)
                param.set("stabilizer.controller", 2)
            await self._sleep_with_live_updates(2.5)

            print("Landing...")
            self._set_state(SwarmState.LANDED)
            stagger_groups = self._stagger_groups(n_drones)
            for group_idx, group in enumerate(stagger_groups):
                await asyncio.gather(
                    *[
                        self._connected_cfs[i].high_level_commander().land(
                            pad_positions[i][2], None, LANDING_DURATION, None
                        )
                        for i in group
                    ]
                )
                await self._sleep_with_live_updates(LANDING_DURATION + 0.5)
            needs_landing = False
            # await self._sleep_with_live_updates(2.5)

            await asyncio.gather(
                *[cf.high_level_commander().stop(None) for cf in self._connected_cfs]
            )
            await asyncio.gather(
                *[cf.platform().send_arming_request(False) for cf in self._connected_cfs]
            )

            self._set_state(SwarmState.CONNECTED)
        except Exception as exc:
            self._set_state(SwarmState.ERROR)
            self._gui.on_error("Flight Failed", f"Flight sequence failed: {exc}")
        finally:
            if needs_landing and self._connected_cfs:
                print("Emergency landing...")
                await asyncio.gather(
                    *[
                        cf.high_level_commander().land(0.0, None, 3.0, None)
                        for cf in self._connected_cfs
                    ],
                    return_exceptions=True,
                )
            await self._stop_live_position_logging()
            self._gui.on_live_mode_stopped()
            if self._connected_cfs:
                self._gui.on_fly_enabled(True)

    # -- Private helpers -----------------------------------------------------

    def _on_connect_task_done(self, task: asyncio.Task) -> None:
        self._connect_task = None
        if task.cancelled():
            return
        _ = task.exception()

    def _on_fly_task_done(self, task: asyncio.Task) -> None:
        self._fly_task = None
        if task.cancelled():
            return
        _ = task.exception()

    async def _disconnect_all(self) -> None:
        if not self._connected_cfs:
            return
        for cf in self._connected_cfs:
            param = cf.param()
            param.set("colorLedBot.wrgb8888", 0x00000000)
        await asyncio.gather(
            *[cf.disconnect() for cf in self._connected_cfs],
            return_exceptions=True,
        )
        self._connected_cfs = []
        self._link_context = None
        self._gui.on_fly_enabled(False)

    @staticmethod
    async def _read_pad_position(cf: object) -> tuple[float, float, float]:
        log = cf.log()
        block = await log.create_block()
        await block.add_variable("stateEstimate.x")
        await block.add_variable("stateEstimate.y")
        await block.add_variable("stateEstimate.z")
        log_stream = await block.start(LOG_INTERVAL)
        try:
            values = (await log_stream.next()).data
            return (
                float(values["stateEstimate.x"]),
                float(values["stateEstimate.y"]),
                float(values["stateEstimate.z"]),
            )
        finally:
            await log_stream.stop()

    async def _start_live_position_logging(self) -> None:
        self._latest_positions = [{"data": None} for _ in self._connected_cfs]
        self._live_log_streams = []

        for cf in self._connected_cfs:
            log = cf.log()
            block = await log.create_block()
            await block.add_variable("stateEstimate.x")
            await block.add_variable("stateEstimate.y")
            await block.add_variable("stateEstimate.z")
            log_stream = await block.start(LOG_INTERVAL)
            self._live_log_streams.append(log_stream)

    async def _stop_live_position_logging(self) -> None:
        for stream in self._live_log_streams:
            stop = getattr(stream, "stop", None)
            if callable(stop):
                result = stop()
                if asyncio.iscoroutine(result):
                    await result
        self._live_log_streams = []
        self._latest_positions = []

    async def _poll_live_positions_once(self, max_wait: float = 0.12) -> None:
        if not self._live_log_streams:
            return

        samples = await asyncio.gather(
            *[
                asyncio.wait_for(stream.next(), timeout=max_wait)
                for stream in self._live_log_streams
            ],
            return_exceptions=True,
        )

        for idx, sample in enumerate(samples):
            if isinstance(sample, Exception):
                continue
            self._latest_positions[idx]["data"] = sample.data

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
        self._gui.on_live_positions(positions)

    async def _sleep_with_live_updates(self, duration: float) -> None:
        if duration <= 0:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + duration
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            step = min(0.12, remaining)
            if self._live_log_streams:
                await self._poll_live_positions_once(max_wait=step)
            else:
                await asyncio.sleep(step)

    # -- CSV / spline helpers ------------------------------------------------

    @staticmethod
    def _stagger_groups(n: int, stride: int = STAGGER_STRIDE) -> list:
        """Return groups of drone indices for staggered launch/land.

        Round 0: [0, stride, 2*stride, ...]
        Round 1: [1, 1+stride, 1+2*stride, ...]
        ...
        """
        groups = []
        for offset in range(stride):
            group = list(range(offset, n, stride))
            if group:
                groups.append(group)
        return groups

    @staticmethod
    def _load_phase_waypoints(
        csv_path: Path,
        n_drones: int,
        prepend_points: list | None = None,
        reverse: bool = False,
    ) -> list:
        df = pd.read_csv(csv_path)
        waypoints_per_drone: list[np.ndarray] = []

        if prepend_points is not None and len(prepend_points) != n_drones:
            raise ValueError(
                f"{csv_path.name} prepend point count ({len(prepend_points)}) "
                f"does not match drone count ({n_drones})"
            )

        for idx in range(n_drones):
            try:
                xs = df[f"x{DRONE_IDX_MAPPING[idx]}"].values
                ys = df[f"y{DRONE_IDX_MAPPING[idx]}"].values
                zs = df[f"z{DRONE_IDX_MAPPING[idx]}"].values
            except KeyError as exc:
                raise ValueError(
                    f"{csv_path.name} is missing columns for drone index {idx} "
                    f"(x{idx}, y{idx}, z{idx})"
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
                raise ValueError(
                    f"{csv_path.name} must contain at least 2 waypoints per drone"
                )
            waypoints_per_drone.append(waypoints)

        return waypoints_per_drone

    @classmethod
    def _load_phase_splines(
        cls,
        csv_path: Path,
        n_drones: int,
        prepend_points: list | None = None,
    ) -> list:
        waypoints_per_drone = cls._load_phase_waypoints(
            csv_path, n_drones, prepend_points=prepend_points
        )
        return [CubicBezierSpline.from_waypoints(wp) for wp in waypoints_per_drone]

    @classmethod
    def _load_phase_splines_reversed(cls, csv_path: Path, n_drones: int) -> list:
        waypoints_per_drone = cls._load_phase_waypoints(csv_path, n_drones, reverse=True)
        return [CubicBezierSpline.from_waypoints(wp) for wp in waypoints_per_drone]

    @classmethod
    def _load_phase_last_points(cls, csv_path: Path, n_drones: int) -> list:
        waypoints_per_drone = cls._load_phase_waypoints(csv_path, n_drones)
        return [wp[-1] for wp in waypoints_per_drone]

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
            CompressedStart(
                spline.beziers[0].p0[0],
                spline.beziers[0].p0[1],
                spline.beziers[0].p0[2],
                0.0,
            ),
            coeffs,
            start_addr=offset,
        )
        print(f"Uploaded trajectory {trajectory_id} ({bytes_written} bytes)")
        await cf.high_level_commander().define_trajectory(
            trajectory_id, offset, len(spline.beziers), 1
        )
        return len(spline.beziers) * segment_duration, bytes_written + 128
