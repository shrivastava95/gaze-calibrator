# What is this data?
The data is sourced from [The DIEM Project](https://thediemproject.wordpress.com/videos-and%c2%a0data/)

# Which files the current script actually needs

Only **two things** are required:
1. `visualize_gaze.py`: the script itself.
2. one gaze file from: `data/samples/arctic_bears_1066x710/event_data/*`. One gaze file contains gaze samples (x,y positions; timestamps; validity flags) recorded from **one participant** watching **one video**.

# The Contents of a Gaze file

- Each `.gze` or `.asc` file stores gaze data from one person watching one video.
- Rows are individual samples containing left/right eye coordinates, timestamps, and validity flags.
- There are nine tab-separated columns: index, left-eye (x, y, t, valid), right-eye (x, y, t, valid).
- Coordinates are in video pixel units.
- Timestamps are in milliseconds from the start of the clip.
- Invalid samples use zeros and must be removed.
- Sampling frequency is roughly 250 Hz.
- Filenames encode session, subject, and video ID.
- `.gze` files are gzip-compressed versions of `.asc` files.
- The data describe how a viewer’s gaze moves across the video frame over time.



# Minimal File Tree

```
your_project/
│
├── INFO.md
│   **YOU ARE HERE**
│
├── visualize_gaze.py
│   # The Python script that loads one DIEM gaze file and visualizes the gaze trajectory.
│
└── data/
    └── samples/
        └── arctic_bears_1066x710/
            │
            ├── video/
            │   └── arctic_bears_1066x710.mp4
            │       # Original DIEM video (not used by the gaze plot script, but part of dataset)
            │
            ├── audio/
            │   └── arctic_bears_1066x710.wav
            │       # Original audio track (not used by the script)
            │
            └── event_data/
                ├── diem5s01.asc_15_arctic_bears_1066x710.txt
                ├── diem5s02.asc_16_arctic_bears_1066x710.txt
                ├── ...
                └── diem5s53.asc_19_arctic_bears_1066x710.txt
                    # All DIEM eye-tracking recordings for this video.
                    # The script uses exactly one of these as input.
```
