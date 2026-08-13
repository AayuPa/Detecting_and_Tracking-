import os

import cv2


def read_video(video_path):
    """Load every frame into memory. Fine for short clips, not for full matches —
    prefer iter_video_frames for anything longer than a few thousand frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path, fps=30):
    if not output_video_frames:
        raise ValueError("No frames to write.")

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    height, width = output_video_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def iter_video_frames(video_path):
    """Yield frames one at a time so memory stays flat regardless of clip length."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def get_video_writer(reference_video_path, output_video_path, fps=None):
    """Build a writer matching the source video's dimensions and frame rate."""
    cap = cv2.VideoCapture(reference_video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open reference video: {reference_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Screen recordings and some containers report 0 or a nonsense fps.
    if fps is None:
        fps = source_fps if source_fps and source_fps > 0 else 30

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Could not open video writer for: {output_video_path}")

    return writer
