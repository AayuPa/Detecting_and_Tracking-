from sympy import true
from  ultralytics import YOLO
model = YOLO('models/best.pt')
results= model.predict("/Users/aayushpawar/Desktop/football_tracker/Screen Recording 2026-07-22 at 10.11.01 PM.mov",save=True)
print(results[0])
print('---------------------------')
for box in results[0].boxes:
	print(box)