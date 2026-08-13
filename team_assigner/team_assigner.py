import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:image.shape[0] // 2, :]

        # frame comes straight from cv2.VideoCapture, so it's BGR
        hsv_image = cv2.cvtColor(top_half_image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        player_mask = green_mask == 0

        player_pixels = top_half_image.reshape(-1, 3)[player_mask.reshape(-1)]
        if len(player_pixels) < 2:
            player_pixels = top_half_image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_pixels)

        labels = kmeans.labels_
        counts = np.bincount(labels)
        jersey_cluster = np.argmax(counts)

        return kmeans.cluster_centers_[jersey_cluster]

    def _darken_for_visibility(self, color, max_value=160):
        # caps HSV brightness so pale colors (e.g. bright yellow) stay
        # readable against the pitch, without changing the hue used for
        # display. Classification uses the original, undarkened centers.
        color_uint8 = np.clip(np.array(color), 0, 255).astype(np.uint8).reshape(1, 1, 3)
        h, s, v = cv2.cvtColor(color_uint8, cv2.COLOR_BGR2HSV)[0, 0]
        v = min(int(v), max_value)
        darker_bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        return tuple(int(c) for c in darker_bgr)

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player in player_detections.items():
            bbox = player["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = self._darken_for_visibility(kmeans.cluster_centers_[0])
        self.team_colors[2] = self._darken_for_visibility(kmeans.cluster_centers_[1])

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        self.player_team_dict[player_id] = team_id

        return team_id
