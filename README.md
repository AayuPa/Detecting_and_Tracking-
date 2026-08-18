# Football Tracker

Computer vision pipeline for amateur football footage. Detects players, goalkeepers,
referees and the ball with a fine-tuned YOLO11m model, associates them across frames
with ByteTrack, separates the two teams by jersey colour, and renders an annotated
video.

<!-- TODO: replace with a real 10-20s clip of annotated output.
     This is the single highest-value thing in the repo. A reviewer watches it
     before reading a line of code. -->
![Demo](assets/demo.gif)

---

## What's implemented

- **Detection.** YOLO11m fine-tuned on 4 classes (`ball`, `goalkeeper`, `player`,
  `referee`), run in batches of 20 frames. Weights ship in `models/best.pt`.
- **Tracking.** ByteTrack via `supervision`, producing per-frame `track_id → bbox`
  dictionaries. Goalkeepers are remapped to the `player` class before tracking so
  ByteTrack sees one coherent class, rather than switching identity when a keeper is
  reclassified between frames.
- **Ball handling.** The highest-confidence ball candidate per frame is kept. At low
  confidence thresholds the detector proposes several.
- **Team assignment.** For each player crop, the top half is masked in HSV to remove
  pitch green, then k-means (k=2) separates jersey from skin and shorts. Per-player
  jersey colours are clustered again across the squad to yield two team colours.
  Assignment is cached per `track_id`, so clustering runs once per new player instead
  of once per frame.
- **Rendering.** Corner-bracket boxes in team colour, yellow circle on the ball, green
  brackets on referees. Output dimensions and frame rate are inherited from the source
  video.
- **Track caching.** Detections can be pickled to a stub, so the annotation code can be
  iterated on without re-running inference.

## Not implemented

Listed because a partial pipeline honestly described is more useful than a roadmap
that reads as if it were finished:

- Ball position interpolation across missed detections
- Pitch homography and field-coordinate mapping
- Event extraction (passes, corners, shots)
- Player speed and distance

---

## Model

| | |
|---|---|
| Backbone | `yolo11m` |
| Classes | `ball`, `goalkeeper`, `player`, `referee` |
| Training resolution | 960px |
| Epochs / batch | 100 / 8 |
| Dataset | Roboflow `football-players-detection-3zvbc`, v19 |
| Trained | July 2026, ultralytics 8.4.104 |

`notebooks/train_detector.ipynb` reproduces this. Note that it needs a GPU runtime; at
960px on CPU, a single epoch runs into the hours.

---

## Setup

```bash
git clone https://github.com/<user>/football-tracker.git
cd football-tracker

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Weights are committed at `models/best.pt` (39MB), so there's no extra download step.

To re-train, set your Roboflow key in the environment rather than in the notebook:

```bash
export ROBOFLOW_API_KEY="your_key_here"
```

---

## Usage

```bash
python main.py --input input_videos/match.mp4 --output output_videos/annotated.mp4
```

Cache detections on the first run, then reuse them while iterating on drawing code:

```bash
python main.py --input input_videos/match.mp4 --stub stubs/match.pkl
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *required* | Source video |
| `--output` | `output_videos/annotated.mp4` | Annotated output |
| `--model` | `models/best.pt` | YOLO weights |
| `--conf` | `0.1` | Detection confidence threshold |
| `--stub` | `None` | Pickle path for cached tracks, written if absent and read if present |
| `--no-stub-read` | off | Force re-detection even if the stub exists |

### Debugging helpers

```bash
python scripts/inspect_video.py input_videos/match.mp4   # dimensions, fps, frame count
python scripts/raw_predict.py input_videos/match.mp4     # detections with no tracking
```

`notebooks/color_assignment.ipynb` walks through the jersey-colour clustering step by
step, including why naive k-means over the whole crop latches onto pitch green and how
the HSV mask fixes it.

---

## Results

<!-- TODO: hand-tag a 200-500 frame segment of your own footage and fill this in.
     Do not estimate. An empty table is neutral; an invented one is disqualifying
     the moment anyone checks. If you're short on time, delete this section.

     Note that validation mAP from the Roboflow split is NOT the number that
     matters here. That split is broadcast footage, and the target domain is
     amateur fixed-camera video. Report performance on your own clips. -->

Measured on a hand-annotated `N`-frame segment of amateur footage:

| Class | Precision | Recall |
|-------|-----------|--------|
| Player | – | – |
| Referee | – | – |
| Ball | – | – |

---

## Known limitations

- **Track IDs are not persistent across a long clip.** A single fixed camera plus
  frequent occlusion means ByteTrack assigns a new ID when a player is lost and
  reacquired. IDs are dependable within a passage of play, not across a half.
- **Domain gap.** The detector was fine-tuned on the Roboflow set, which is broadcast
  footage. Amateur fixed-camera video differs in resolution, camera height and colour,
  so expect degradation and validate on target footage before quoting numbers.
- **`conf=0.1` is deliberately permissive and not yet tuned.** It buys recall on the
  ball at the cost of false positives elsewhere. The highest-confidence-ball rule
  treats the symptom; the threshold itself still needs a sweep against ground truth.
- **Team assignment is fitted once**, on the first frame containing two or more
  players, and cached per track ID. A mid-clip lighting change, or a first frame
  containing players from only one team, will skew the clusters. It also has no
  concept of goalkeepers or referees wearing a third colour.
- **Jersey-number OCR is out of scope.** At amateur footage resolution the digits are
  typically under 15px tall and not recoverable.
- **`read_video` loads every frame into memory** and is unsuitable for full matches.
  The main path uses the streaming `iter_video_frames` instead.

---

## Project structure

```
football-tracker/
├── main.py                        # CLI entrypoint
├── tracker/
│   └── tracker.py                 # YOLO inference, ByteTrack association, drawing
├── team_assigner/
│   └── team_assigner.py           # HSV pitch masking + two-stage k-means
├── utils/
│   ├── video_utils.py             # streaming frame reader, writer construction
│   └── bbox_util.py               # bbox geometry helpers
├── scripts/
│   ├── inspect_video.py           # video property dump
│   └── raw_predict.py             # detector-only debugging
├── notebooks/
│   ├── train_detector.ipynb       # Roboflow download + YOLO11m fine-tune
│   └── color_assignment.ipynb     # jersey colour clustering walkthrough
├── models/best.pt                 # fine-tuned detector
├── stubs/                         # cached detections (gitignored)
├── input_videos/                  # gitignored
└── output_videos/                 # gitignored
```

---

## Notes

`supervision` is pinned below `0.30`. The `sv.ByteTrack` interface used here is
deprecated in newer releases, and unpinning will require migrating the tracker call.
