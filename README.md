# Gaze-Driven Motion-Compensated Video Generator

This repository contains a CLI tool that follows the DIEM specification provided in the prompt. The tool parses a single DIEM gaze trace, aligns it to the video timeline, and renders a gaze-locked, motion-compensated output video with padded borders.

## Features

- Right-eye–first gaze selection with automatic left-eye fallback
- Timestamp sorting and nearest-sample lookup for every frame
- Output canvas scaled (default 4×) with configurable padding color
- Black padding when frame shifts expose borders
- H.264 output via FFmpeg with optional audio pass-through from the source video

## Requirements

1. Python 3.10+
2. FFmpeg/FFprobe available on your `PATH`
3. Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python gaze_motion_compensator.py \
  --video /home/kadda/Desktop/personal/gaze-calibrator/data/samples/arctic_bears_1066x710/video/arctic_bears_1066x710.mp4 \
  --gaze /home/kadda/Desktop/personal/gaze-calibrator/data/samples/arctic_bears_1066x710/event_data/diem5s01.asc_15_arctic_bears_1066x710.txt \
  --output /home/kadda/Desktop/personal/gaze-calibrator/output/arctic_bears_1066x710_diem5s01.mp4
```

Optional arguments:

- `--scale` (default `2`): multiplies the base resolution before padding
- `--padding` (default `black`): accepts `black`, `white`, `gray`, or a `#RRGGBB` hex color
- `--include-events`: choose which DIEM event codes to keep (e.g., `fixation`, `saccade`)
- `--max-frames`: process only the first N frames (handy for quick tests)
- `--smoothing`: `kalman` (default) or `none`
- `--smoothing-process-var` / `--smoothing-measurement-var`: tune how snappy vs. smooth the lock should feel
- `--center-marker-radius` / `--center-marker-thickness`: tweak the red focal circle drawn at the canvas center
- `--codec`, `--crf`, `--preset`: forwarded to FFmpeg for video encoding control

## Implementation Notes

- Frames are decoded with OpenCV, shifted so the gaze coordinate lands exactly in the center of the enlarged canvas, and streamed to FFmpeg for encoding.
- When the video lacks an audio stream, the script logs a warning and outputs a silent gaze-locked video; otherwise, it copies the original audio track losslessly.
- The script enforces even output dimensions so the final H.264 bitstream remains compatible with `yuv420p`.

## Collating and Visualizing All Gaze Tracks

Use `gaze_collator.py` to read every DIEM gaze file in a directory, aggregate them across subjects, and generate a Matplotlib heatmap plus a timeline plot:

```bash
python gaze_collator.py \
  --video /home/kadda/Desktop/personal/gaze-calibrator/data/samples/arctic_bears_1066x710/video/arctic_bears_1066x710.mp4 \
  --gaze-dir /home/kadda/Desktop/personal/gaze-calibrator/data/samples/arctic_bears_1066x710/event_data \
  --figure /home/kadda/Desktop/personal/gaze-calibrator/output/arctic_bears_1066x710_gaze_summary.png \
  --include-events fixation saccade \
  --bins 96
```

The collator:

- Detects FPS/resolution from the video to convert DIEM frame indices into timestamps
- Parses every left/right eye sample, filtering by the event types you specify (default: fixation + saccade)
- Builds a 2D spatial heatmap to show where viewers looked most often
- Plots sample density over time, so you can see participation coverage across the clip

