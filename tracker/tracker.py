from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2

sys.path.append("../")
from utils import get_bbox_width, get_center_of_bbox, iter_video_frames

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def get_object_tracks(self, video_path, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        batch_size = 20
        batch = []

        def process_batch(frames_batch):
            batch_detections = self.model.predict(frames_batch, conf=0.1)
            for detection in batch_detections:
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}

                detection_supervision = sv.Detections.from_ultralytics(detection)

                # Goalkeeper -> player
                for object_id, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_supervision.class_id[object_id] = cls_names_inv["player"]

                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})
                frame_num = len(tracks["players"]) - 1

                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    if cls_id == cls_names_inv["player"]:
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    elif cls_id == cls_names_inv["referee"]:
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]

                    if cls_id == cls_names_inv["ball"]:
                        tracks["ball"][frame_num][1] = {"bbox": bbox}

        for frame in iter_video_frames(video_path):
            batch.append(frame)
            if len(batch) == batch_size:
                process_batch(batch)
                batch = []

        if batch:
            process_batch(batch)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_corner_box(self, frame, bbox, color, track_id=None):

        x1, y1, x2, y2 = map(int, bbox)

        corner_length = 20
        thickness = 5

        # ---------- TOP LEFT ----------
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)

        # ---------- TOP RIGHT ----------
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)

        # ---------- BOTTOM LEFT ----------
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)

        # ---------- BOTTOM RIGHT ----------
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
        return frame

    def draw_ball(self, frame, bbox):

        x1, y1, x2, y2 = map(int, bbox)

        center = (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

        radius = max(x2 - x1, y2 - y1) // 2

        cv2.circle(
            frame,
            center,
            radius,
            (0, 255, 255),
            2
        )

        return frame

    def draw_annotations_on_frame(self, frame, frame_num, tracks):
        frame = frame.copy()

        player_dict = tracks["players"][frame_num]
        referee_dict = tracks["referees"][frame_num]
        ball_dict = tracks["ball"][frame_num]

        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 225))
            frame = self.draw_corner_box(frame, player["bbox"], color, track_id)
        for _, ball in ball_dict.items():
            frame = self.draw_ball(frame, ball["bbox"])
        for _, referee in referee_dict.items():
            frame = self.draw_corner_box(frame, referee["bbox"], (0, 225, 0))

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            output_video_frames.append(self.draw_annotations_on_frame(frame, frame_num, tracks))
        return output_video_frames