"""Run raw YOLO prediction with no tracking and dump per-box detections.

Debugging aid for checking what the detector alone sees, before ByteTrack
association muddies the picture.
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--save", action="store_true", help="Write annotated output to runs/")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.predict(args.video, conf=args.conf, save=args.save)

    first = results[0]
    print("Classes:", first.names)
    print("-" * 40)
    for box in first.boxes:
        cls_id = int(box.cls[0])
        print(f"{first.names[cls_id]:<12} conf={float(box.conf[0]):.3f} xyxy={box.xyxy[0].tolist()}")


if __name__ == "__main__":
    main()
