import cv2
from tracker import Tracker
from utils import iter_video_frames, get_video_writer
from team_assigner import TeamAssigner

def main():
    video_path = "/Users/aayushpawar/Desktop/New Folder With Items/Vids/REC-20260725193942.mp4"
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(
        video_path,
        read_from_stub=True,
        stub_path="stubs/linkedin_post_stub.pkl"
    )

    num_tracked_frames = len(tracks["players"])

    # team colors need a frame with >=2 players to cluster into 2 teams;
    # find the first such frame from the tracks data (no video decode needed)
    team_colors_frame_num = next(
        (i for i in range(num_tracked_frames) if len(tracks["players"][i]) >= 2),
        None
    )

    team_assigner = TeamAssigner()
    if team_colors_frame_num is not None:
        for frame_num, frame in enumerate(iter_video_frames(video_path)):
            if frame_num == team_colors_frame_num:
                team_assigner.assign_team_color(frame, tracks["players"][frame_num])
                break

    writer = get_video_writer(video_path, "output_videos/LinkedIn_Post.mp4")
    cropped_saved = False

    for frame_num, frame in enumerate(iter_video_frames(video_path)):
        if frame_num >= num_tracked_frames:
            break

        if team_assigner.kmeans is not None:
            for track_id, player in tracks["players"][frame_num].items():
                team_id = team_assigner.get_player_team(frame, player["bbox"], track_id)
                player["team_color"] = tuple(int(c) for c in team_assigner.team_colors[team_id])

        annotated_frame = tracker.draw_annotations_on_frame(frame, frame_num, tracks)
        writer.write(annotated_frame)

        if not cropped_saved:
            for track_id, player in tracks["players"][frame_num].items():
                bbox = player["bbox"]
                cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                cv2.imwrite("output_image/cropped_image.jpg", cropped_image)
                cropped_saved = True
                break

    writer.release()

    if team_assigner.kmeans is not None:
        print("Team colors:", team_assigner.team_colors)

if __name__ == "__main__":
    main()
