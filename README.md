# Football Tracker

Computer vision pipeline for amateur full-match football footage. Detects and tracks
players, goalkeepers, referees, and the ball across frames, and renders an annotated
output video.

<!-- TODO: Replace with a real 10–20s clip of annotated output. This is the single
     highest-value thing in the README — put it above the fold. -->
![Demo](assets/demo.gif)

---

## What it does

- **Detection** — YOLOv8 (`ultralytics`) fine-tuned on football footage, 4 classes:
  player, goalkeeper, referee, ball
- **Tracking** — ByteTrack (`supervision`) for frame-to-frame identity association
- **Annotation** — OpenCV overlay of bounding ellipses, track IDs, and ball marker

<!-- TODO: Delete the bullets below that aren't actually implemented yet.
     An honest short list beats an aspirational long one. -->
- Team separation via k-means clustering on jersey/torso crops
- Ball position interpolation across missed detections
- Pitch homography for pixel → field coordinate mapping
- Rule-based event extraction (passes, corners, shots) exported as JSON

---

## Results

<!-- TODO: Hand-tag one 200–500 frame segment as ground truth and fill this in
     with measured numbers. Do not estimate. A real table with mediocre numbers
     is far more credible than a vague claim of "high accuracy". -->

Measured on a hand-annotated `N`-frame segment of amateur match footage:

| Class      | Precision | Recall | Notes |
|------------|-----------|--------|-------|
| Player     | –         | –      |       |
| Goalkeeper | –         | –      |       |
| Referee    | –         | –      |       |
| Ball       | –         | –      |       |

Detection confidence threshold: `conf=0.__`
<!-- TODO: state the value you actually settled on and one line on why -->

---

## Known limitations

Stated up front, because they're inherent to the setup rather than bugs:

- **Track IDs are not persistent across long clips.** Single fixed camera plus
  frequent occlusion means ByteTrack reassigns IDs after a player is lost. IDs are
  reliable within a possession, not across a half.
- **Jersey-number OCR is out of scope.** At amateur footage resolution the digits
  are typically under 15px tall — not recoverable. Player identity would need a
  second camera or manual initialization.
- **Ball detection is intermittent.** Small, fast, and frequently occluded; expect
  gaps that require interpolation rather than continuous detection.
- <!-- TODO: add anything else you hit. Reviewers trust a README more when it
     names its own weaknesses. -->

---

## Setup

```bash
git clone https://github.com/<user>/football_tracker.git
cd football_tracker

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Model weights

<!-- TODO: pick one and delete the other -->

**If committed:** weights are at `models/best.pt`, no extra step needed.

**If shipped via Release:**
```bash
mkdir -p models
# download best.pt from the Releases page and place it in models/
```

---

## Usage

```bash
python main.py --input input_videos/match.mp4 --output output_videos/annotated.mp4
```

<!-- TODO: match these to your actual argparse flags, or replace this block with
     the real invocation if paths are currently hardcoded. -->

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | – | Path to source video |
| `--output` | – | Path for annotated output |
| `--conf` | – | Detection confidence threshold |

---

## Project structure

```
football_tracker/
├── detector.py      # YOLOv8 wrapper — batched frame inference
├── tracker.py       # ByteTrack association, per-class track assembly
├── visualize.py     # OpenCV annotation and video writing
├── main.py          # Pipeline entrypoint
├── models/
│   └── best.pt      # Fine-tuned detection weights
├── input_videos/    # gitignored
├── output_videos/   # gitignored
└── requirements.txt
```

---

## Notes on dependencies

`supervision` is pinned below `0.30` — the `sv.ByteTrack` interface used here is
deprecated in newer releases. Unpinning will require migrating the tracker call.

---

## Roadmap

<!-- TODO: keep this short and only list things you'd actually build next.
     A five-item roadmap reads as a plan; a fifteen-item one reads as a wish list. -->

- [ ] Ball interpolation across detection gaps
- [ ] Team assignment by jersey color clustering
- [ ] Pitch homography → field coordinates
- [ ] Pass/event detection exported as JSON
