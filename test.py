import cv2

cap = cv2.VideoCapture("videos/Screen Recording 2026-07-22 at 10.11.01 PM.mov")

print("Width :", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("Height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("FPS   :", cap.get(cv2.CAP_PROP_FPS))
print("Frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

cap.release()