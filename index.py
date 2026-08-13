import argparse
import os

from tracker import Tracker
from team_assigner import TeamAssigner
from utils import iter_video_frames, get_video_writer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect, track and annotate players, referees and the ball in football footage."
    )
    parser.add_argument("--input", required=True, help="Path to the source video")
    parser.add_argument(
        "--output",
        default="output_videos/annotated.mp4",
        help="Path for the annotated output video",
    )
    parser.add_argument("--model", default="models/best.pt", help="Path to the YOLO weights")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help=(
            "Detection confidence threshold. 0.1 is permissive and produces "
            "spurious ball candidates; raise it if you see false positives."
        ),
    )
    parser.add_argument(
        "--stub",
        default=None,
        help="Path to a pickle of cached tracks. Written if absent, read if present.",
    )
    parser.add_argument(
        "--no-stub-read",
        action="store_true",
        help="Ignore an existing stub and re-run detection",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tracker = Tracker(args.model, conf=args.conf)
    tracks = tracker.get_object_tracks(
        args.input,
        read_from_stub=not args.no_stub_read,
        stub_path=args.stub,
    )

    num_tracked_frames = len(tracks["players"])
    if num_tracked_frames == 0:
        raise SystemExit("No frames were tracked — check the input path and model weights.")

    team_assigner = TeamAssigner()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = get_video_writer(args.input, args.output)

    try:
        for frame_num, frame in enumerate(iter_video_frames(args.input)):
            if frame_num >= num_tracked_frames:
                break

            players = tracks["players"][frame_num]

            # Fit the two-team colour model on the first frame that actually
            # contains enough players to cluster. Done inline so the video is
            # decoded only once.
            if team_assigner.kmeans is None and len(players) >= 2:
                team_assigner.assign_team_color(frame, players)

            if team_assigner.kmeans is not None:
                for track_id, player in players.items():
                    team_id = team_assigner.get_player_team(frame, player["bbox"], track_id)
                    player["team_color"] = team_assigner.team_colors[team_id]

            writer.write(tracker.draw_annotations_on_frame(frame, frame_num, tracks))
    finally:
        writer.release()

    if team_assigner.kmeans is not None:
        print("Team colours (BGR):", team_assigner.team_colors)
    else:
        print("Team colours not assigned — no frame contained two or more players.")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
