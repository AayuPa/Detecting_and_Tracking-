import cv2
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    print("Opened:", cap.isOpened())
    frames=[]
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(output_video_path, fourcc, 30, (output_video_frames[0].shape[1],output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def iter_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()

def get_video_writer(reference_video_path, output_video_path, fps=None):

    cap = cv2.VideoCapture(reference_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_video_path, fourcc, video_fps, (width, height))

