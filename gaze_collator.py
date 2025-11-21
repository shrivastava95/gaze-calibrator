#!/usr/bin/env python3
"""
Collate DIEM gaze files across all subjects and build summary visualizations.

Example:
    python gaze_collator.py \
        --video /home/kadda/Desktop/personal/gaze-calibrator/data/samples/arctic_bears_1066x710/video/arctic_bears_1066x710.mp4 \
        --gaze-dir /home/kadda/Desktop/personal/gaze-calibrator/data/samples/arctic_bears_1066x710/event_data \
        --figure /home/kadda/Desktop/personal/gaze-calibrator/output/arctic_heatmap.png
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib
import numpy as np

# Headless backend for servers/CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVENT_MAP = {
    "blink": 0,
    "fixation": 1,
    "saccade": 2,
}
EVENT_LABELS = {v: k for k, v in EVENT_MAP.items()}


@dataclass(frozen=True)
class GazeRecord:
    subject: str
    eye: str
    frame_idx: int
    timestamp_ms: float
    x: float
    y: float
    event_code: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DIEM gaze files and visualize their coverage.")
    parser.add_argument("--video", required=True, type=Path, help="Path to the DIEM MP4 (used for fps/resolution).")
    parser.add_argument("--gaze-dir", required=True, type=Path, help="Directory containing diem*.txt gaze files.")
    parser.add_argument(
        "--figure",
        required=True,
        type=Path,
        help="Output path for the Matplotlib figure (PNG, PDF, etc.).",
    )
    parser.add_argument(
        "--include-events",
        nargs="+",
        default=["fixation", "saccade"],
        choices=sorted(EVENT_MAP.keys()),
        help="Eye-event categories to include when collating samples (default: fixation + saccade).",
    )
    parser.add_argument(
        "--bins",
        default=80,
        type=int,
        help="Number of bins per axis for the 2D heatmap (default: 80).",
    )
    parser.add_argument(
        "--min-boundary",
        default=0.0,
        type=float,
        help="Minimum allowed coordinate (useful if you need to clip negative samples).",
    )
    parser.add_argument(
        "--max-x",
        type=float,
        default=None,
        help="Override max X bound; defaults to the video width.",
    )
    parser.add_argument(
        "--max-y",
        type=float,
        default=None,
        help="Override max Y bound; defaults to the video height.",
    )
    return parser.parse_args()


def detect_video_geometry(video_path: Path) -> Tuple[float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0:
        raise RuntimeError("Video FPS was zero or undefined.")
    logging.info("Video geometry: %dx%d @ %.3f fps", width, height, fps)
    return fps, width, height


def iter_gaze_files(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        raise RuntimeError(f"{directory} is not a directory.")
    for path in sorted(directory.glob("*.txt")):
        yield path


def parse_gaze_file(
    path: Path,
    fps: float,
    width: float,
    height: float,
    allowed_events: Sequence[int],
    min_boundary: float,
) -> List[GazeRecord]:
    records: List[GazeRecord] = []
    allowed_set = set(allowed_events)
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

            timestamp_ms = (frame_idx - 1) * ms_per_frame

            def append_eye(eye_label: str, x: float, y: float, event_code: int) -> None:
                if event_code not in allowed_set:
                    return
                if not (min_boundary <= x <= width) or not (min_boundary <= y <= height):
                    return
                records.append(
                    GazeRecord(
                        subject=path.stem,
                        eye=eye_label,
                        frame_idx=frame_idx,
                        timestamp_ms=timestamp_ms,
                        x=x,
                        y=y,
                        event_code=event_code,
                    )
                )

            append_eye("left", left_x, left_y, left_event)
            append_eye("right", right_x, right_y, right_event)
    logging.info("Parsed %d samples from %s", len(records), path.name)
    return records


def collate_records(
    gaze_dir: Path,
    fps: float,
    width: float,
    height: float,
    allowed_events: Sequence[int],
    min_boundary: float,
) -> List[GazeRecord]:
    all_records: List[GazeRecord] = []
    for path in iter_gaze_files(gaze_dir):
        all_records.extend(parse_gaze_file(path, fps, width, height, allowed_events, min_boundary))
    logging.info("Total samples gathered: %d", len(all_records))
    return all_records


def summarize_records(records: Sequence[GazeRecord]) -> Dict[str, Counter]:
    by_subject: Counter[str] = Counter()
    by_event: Counter[int] = Counter()
    for record in records:
        by_subject[record.subject] += 1
        by_event[record.event_code] += 1
    return {"subjects": by_subject, "events": by_event}


def build_heatmap(records: Sequence[GazeRecord], width: float, height: float, bins: int) -> np.ndarray:
    xs = np.array([rec.x for rec in records], dtype=np.float32)
    ys = np.array([rec.y for rec in records], dtype=np.float32)
    heatmap, xedges, yedges = np.histogram2d(
        xs,
        ys,
        bins=[bins, bins],
        range=[[0, width], [0, height]],
    )
    return heatmap.T, xedges, yedges


def build_timeline(records: Sequence[GazeRecord]) -> Tuple[np.ndarray, np.ndarray]:
    counts = Counter(rec.frame_idx for rec in records)
    frames = np.array(sorted(counts.keys()), dtype=np.int32)
    values = np.array([counts[idx] for idx in frames], dtype=np.int32)
    return frames, values


def plot_results(
    records: Sequence[GazeRecord],
    width: float,
    height: float,
    bins: int,
    figure_path: Path,
) -> None:
    if not records:
        raise RuntimeError("No gaze records available for plotting.")

    heatmap, xedges, yedges = build_heatmap(records, width, height, bins)
    frames, frame_counts = build_timeline(records)

    fig, (ax_heat, ax_time) = plt.subplots(1, 2, figsize=(14, 6))

    heat = ax_heat.imshow(
        heatmap,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="magma",
        aspect="auto",
    )
    ax_heat.set_title("Aggregated gaze density")
    ax_heat.set_xlabel("x (pixels)")
    ax_heat.set_ylabel("y (pixels)")
    fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04, label="sample count")

    ax_time.plot(frames, frame_counts, color="#1976d2")
    ax_time.set_title("Samples per frame across all subjects")
    ax_time.set_xlabel("Frame index")
    ax_time.set_ylabel("Sample count")
    ax_time.grid(True, alpha=0.3)

    fig.suptitle("DIEM gaze aggregation")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    logging.info("Saved visualization to %s", figure_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    fps, width, height = detect_video_geometry(args.video.resolve())
    max_x = args.max_x if args.max_x is not None else width
    max_y = args.max_y if args.max_y is not None else height

    allowed_events = [EVENT_MAP[name] for name in args.include_events]
    records = collate_records(
        gaze_dir=args.gaze_dir.resolve(),
        fps=fps,
        width=max_x,
        height=max_y,
        allowed_events=allowed_events,
        min_boundary=args.min_boundary,
    )
    summary = summarize_records(records)

    logging.info("Subjects with most samples: %s", summary["subjects"].most_common(5))
    logging.info(
        "Event counts: %s",
        {EVENT_LABELS.get(code, str(code)): count for code, count in summary["events"].items()},
    )

    plot_results(records, max_x, max_y, args.bins, args.figure.resolve())


if __name__ == "__main__":
    main()

