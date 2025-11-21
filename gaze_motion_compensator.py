#!/usr/bin/env python3
"""
Gaze-driven motion-compensated video generator for DIEM samples.

Usage example:
    python gaze_motion_compensator.py \
        --video data/samples/arctic_bears_1066x710/video/arctic_bears_1066x710.mp4 \
        --gaze data/samples/arctic_bears_1066x710/event_data/diem5s01.asc_15_arctic_bears_1066x710.txt \
        --output output/arctic_bears_1066x710_gaze_locked.mp4
"""
from __future__ import annotations

import argparse
import logging
import math
import subprocess
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


EVENT_MAP = {
    "blink": 0,
    "fixation": 1,
    "saccade": 2,
}
EVENT_LABELS = {v: k for k, v in EVENT_MAP.items()}


@dataclass(frozen=True)
class GazeSample:
    timestamp_ms: float
    x: float
    y: float


class GazeSmoother:
    def smooth(self, timestamp_ms: float, x: float, y: float) -> Tuple[float, float]:
        raise NotImplementedError


class NoOpSmoother(GazeSmoother):
    def smooth(self, timestamp_ms: float, x: float, y: float) -> Tuple[float, float]:
        return x, y


class KalmanSmoother(GazeSmoother):
    """
    Constant-velocity 2D Kalman filter.
    State vector: [x, y, vx, vy]
    """

    def __init__(self, process_var: float, measurement_var: float):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self._state: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._last_time: Optional[float] = None

    def smooth(self, timestamp_ms: float, x: float, y: float) -> Tuple[float, float]:
        t = timestamp_ms / 1000.0
        measurement = np.array([[x], [y]], dtype=np.float64)
        if self._state is None:
            self._state = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)
            self._covariance = np.eye(4, dtype=np.float64) * 1e3
            self._last_time = t
            return x, y

        dt = max(t - (self._last_time or t), 1e-3)
        self._last_time = t
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q = self.process_var
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        Q = np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        ) * q

        # Predict
        self._state = F @ self._state
        self._covariance = F @ self._covariance @ F.T + Q

        # Update
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        R = np.eye(2, dtype=np.float64) * self.measurement_var
        predicted_measurement = H @ self._state
        y_residual = measurement - predicted_measurement
        S_cov = H @ self._covariance @ H.T + R
        K = self._covariance @ H.T @ np.linalg.inv(S_cov)
        self._state = self._state + K @ y_residual
        identity = np.eye(4, dtype=np.float64)
        self._covariance = (identity - K @ H) @ self._covariance

        return float(self._state[0, 0]), float(self._state[1, 0])


class GazeSeries:
    def __init__(self, samples: Sequence[GazeSample]):
        if not samples:
            raise ValueError("No valid gaze samples were found.")
        self._samples: List[GazeSample] = sorted(samples, key=lambda s: s.timestamp_ms)
        self._timestamps: List[float] = [sample.timestamp_ms for sample in self._samples]

    def nearest(self, timestamp_ms: float) -> GazeSample:
        idx = bisect_left(self._timestamps, timestamp_ms)
        if idx <= 0:
            return self._samples[0]
        if idx >= len(self._samples):
            return self._samples[-1]
        prev_sample = self._samples[idx - 1]
        next_sample = self._samples[idx]
        if timestamp_ms - prev_sample.timestamp_ms <= next_sample.timestamp_ms - timestamp_ms:
            return prev_sample
        return next_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate motion-compensated, gaze-locked output video.")
    parser.add_argument("--video", required=True, type=Path, help="Path to source DIEM video (MP4).")
    parser.add_argument("--gaze", required=True, type=Path, help="Path to DIEM gaze file (txt).")
    parser.add_argument("--output", required=True, type=Path, help="Destination path for encoded MP4.")
    parser.add_argument("--scale", default=2.0, type=float, help="Output scaling factor relative to source resolution.")
    parser.add_argument("--padding", default="black", help="Padding color ('black', 'white', '#RRGGBB', etc.).")
    parser.add_argument("--codec", default="libx264", help="FFmpeg video codec (default: libx264).")
    parser.add_argument("--crf", default=18, type=int, help="FFmpeg CRF for video quality (lower = higher quality).")
    parser.add_argument("--preset", default="medium", help="FFmpeg preset (ultrafast...placebo).")
    parser.add_argument(
        "--include-events",
        nargs="+",
        default=["fixation", "saccade"],
        choices=sorted(EVENT_MAP.keys()),
        help="Which DIEM eye-event codes to keep (default: fixation + saccade).",
    )
    parser.add_argument(
        "--min-boundary",
        default=0.0,
        type=float,
        help="Lower bound for valid coordinates (default: 0).",
    )
    parser.add_argument(
        "--max-x",
        type=float,
        default=None,
        help="Upper X bound override (defaults to the video width).",
    )
    parser.add_argument(
        "--max-y",
        type=float,
        default=None,
        help="Upper Y bound override (defaults to the video height).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit processing to the first N frames (default: entire video).",
    )
    parser.add_argument(
        "--smoothing",
        choices=["kalman", "none"],
        default="kalman",
        help="Smoothing filter to apply to gaze samples (default: kalman).",
    )
    parser.add_argument(
        "--smoothing-process-var",
        type=float,
        default=75.0,
        help="Process variance for the Kalman smoother (higher = faster response).",
    )
    parser.add_argument(
        "--smoothing-measurement-var",
        type=float,
        default=25.0,
        help="Measurement variance for the Kalman smoother (higher = more smoothing).",
    )
    parser.add_argument(
        "--center-marker-radius",
        type=int,
        default=60,
        help="Radius (in pixels) of the center marker circle (default: 60).",
    )
    parser.add_argument(
        "--center-marker-thickness",
        type=int,
        default=2,
        help="Thickness of the center marker outline (default: 2).",
    )
    return parser.parse_args()


def parse_padding_color(value: str) -> Tuple[int, int, int]:
    named = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
    }
    lowercase = value.lower()
    if lowercase in named:
        r, g, b = named[lowercase]
    elif value.startswith("#") and len(value) == 7:
        r = int(value[1:3], 16)
        g = int(value[3:5], 16)
        b = int(value[5:7], 16)
    else:
        raise ValueError(f"Unsupported padding color '{value}'. Use a named color or #RRGGBB.")
    return (b, g, r)


def load_gaze_samples(
    path: Path,
    fps: float,
    max_x: float,
    max_y: float,
    include_events: Sequence[str],
    min_boundary: float,
) -> GazeSeries:
    allowed_events = {EVENT_MAP[name] for name in include_events}
    if not allowed_events:
        raise ValueError("No allowed events specified.")
    samples: List[GazeSample] = []
    ms_per_frame = 1000.0 / fps
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 9:
                continue
            try:
                frame_idx = int(float(parts[0]))
                left_x = float(parts[1])
                left_y = float(parts[2])
                left_event = int(float(parts[4]))
                right_x = float(parts[5])
                right_y = float(parts[6])
                right_event = int(float(parts[8]))
            except ValueError:
                continue

            def is_valid(event_code: int, x: float, y: float) -> bool:
                return (
                    event_code in allowed_events
                    and min_boundary <= x <= max_x
                    and min_boundary <= y <= max_y
                )

            timestamp_ms = (frame_idx - 1) * ms_per_frame

            if is_valid(right_event, right_x, right_y):
                samples.append(GazeSample(timestamp_ms, right_x, right_y))
            elif is_valid(left_event, left_x, left_y):
                samples.append(GazeSample(timestamp_ms, left_x, left_y))

    logging.info(
        "Loaded %d gaze samples from %s using events %s",
        len(samples),
        path,
        [EVENT_LABELS.get(EVENT_MAP[name], name) for name in include_events],
    )
    return GazeSeries(samples)


def detect_audio_stream(source: Path) -> bool:
    """Return True if FFprobe detects at least one audio stream."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(source),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe not found. Install ffmpeg to continue.") from exc
    return bool(result.stdout.strip())


def clamp_scaling_dimension(value: int, scale: float) -> int:
    scaled = int(round(value * scale))
    if scaled % 2:
        scaled += 1
    return max(2, scaled)


def composite_frame(
    frame: np.ndarray,
    canvas_shape: Tuple[int, int, int],
    offset_top: int,
    offset_left: int,
    padding_color: Tuple[int, int, int],
) -> np.ndarray:
    canvas = np.full(canvas_shape, padding_color, dtype=np.uint8)
    src_h, src_w = frame.shape[:2]
    out_h, out_w = canvas_shape[:2]

    dest_top = max(0, offset_top)
    dest_left = max(0, offset_left)
    src_top = max(0, -offset_top)
    src_left = max(0, -offset_left)

    dest_bottom = min(out_h, offset_top + src_h)
    dest_right = min(out_w, offset_left + src_w)

    copy_h = dest_bottom - dest_top
    copy_w = dest_right - dest_left
    if copy_h <= 0 or copy_w <= 0:
        return canvas

    canvas[dest_top : dest_top + copy_h, dest_left : dest_left + copy_w] = frame[
        src_top : src_top + copy_h, src_left : src_left + copy_w
    ]
    return canvas


def build_ffmpeg_process(
    output_path: Path,
    out_width: int,
    out_height: int,
    fps: float,
    codec: str,
    crf: int,
    preset: str,
    audio_source: Optional[Path],
) -> subprocess.Popen:
    size_arg = f"{out_width}x{out_height}"
    cmd: List[str] = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        size_arg,
        "-r",
        f"{fps:.06f}",
        "-i",
        "-",
    ]
    if audio_source is not None:
        cmd.extend(
            [
                "-i",
                str(audio_source),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:a",
                "copy",
                "-shortest",
            ]
        )
    else:
        cmd.extend(["-map", "0:v:0"])

    cmd.extend(
        [
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            "-preset",
            preset,
            "-crf",
            str(crf),
            str(output_path),
        ]
    )

    logging.debug("Launching ffmpeg: %s", " ".join(cmd))
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Install ffmpeg to continue.") from exc
    assert process.stdin is not None
    return process


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    video_path = args.video.resolve()
    gaze_path = args.gaze.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isclose(fps, 0.0):
        raise RuntimeError("Unable to read FPS from video metadata.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info("Video resolution: %dx%d @ %.3f fps", width, height, fps)
    cap.release()

    max_x = args.max_x if args.max_x is not None else width
    max_y = args.max_y if args.max_y is not None else height

    gaze_series = load_gaze_samples(
        gaze_path,
        fps=fps,
        max_x=max_x,
        max_y=max_y,
        include_events=args.include_events,
        min_boundary=args.min_boundary,
    )

    out_width = clamp_scaling_dimension(width, args.scale)
    out_height = clamp_scaling_dimension(height, args.scale)
    output_center_x = out_width / 2.0
    output_center_y = out_height / 2.0
    padding_color = parse_padding_color(args.padding)
    canvas_shape = (out_height, out_width, 3)

    has_audio = detect_audio_stream(video_path)
    if not has_audio:
        logging.warning("No audio stream detected in %s. Output will be silent.", video_path)
    ffmpeg_process = build_ffmpeg_process(
        output_path=output_path,
        out_width=out_width,
        out_height=out_height,
        fps=fps,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
        audio_source=video_path if has_audio else None,
    )
    assert ffmpeg_process.stdin is not None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to reopen video {video_path} for decoding.")

    frame_interval_ms = 1000.0 / fps
    frame_index = 0
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None
    if args.smoothing == "kalman":
        smoother: GazeSmoother = KalmanSmoother(
            process_var=args.smoothing_process_var,
            measurement_var=args.smoothing_measurement_var,
        )
    else:
        smoother = NoOpSmoother()

    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                logging.info("Reached frame limit (%d). Stopping early.", max_frames)
                break
            ok, frame = cap.read()
            if not ok:
                logging.info("Video decoding finished after %d frames.", frame_index)
                break

            frame_time_ms = frame_index * frame_interval_ms
            gaze_sample = gaze_series.nearest(frame_time_ms)
            smooth_x, smooth_y = smoother.smooth(frame_time_ms, gaze_sample.x, gaze_sample.y)
            offset_left = int(round(output_center_x - smooth_x))
            offset_top = int(round(output_center_y - smooth_y))
            composed = composite_frame(frame, canvas_shape, offset_top, offset_left, padding_color)
            cv2.circle(
                composed,
                (int(round(output_center_x)), int(round(output_center_y))),
                args.center_marker_radius,
                (0, 0, 255),
                args.center_marker_thickness,
                lineType=cv2.LINE_AA,
            )
            ffmpeg_process.stdin.write(composed.tobytes())
            frame_index += 1
            if frame_index % 300 == 0:
                logging.info("Processed %d frames", frame_index)
    finally:
        cap.release()
        ffmpeg_process.stdin.close()

    return_code = ffmpeg_process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with non-zero status {return_code}")

    logging.info("Done. Wrote %s (%dx%d)", output_path, out_width, out_height)


if __name__ == "__main__":
    main()

