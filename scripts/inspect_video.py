"""Print basic properties of a video file. Useful for spotting screen recordings
that report a bogus frame rate or frame count."""
import argparse

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to the video file")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open: {args.video}")

    print("Width :", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("FPS   :", cap.get(cv2.CAP_PROP_FPS))
    print("Frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    cap.release()


if __name__ == "__main__":
    main()
