import cv2
import mediapipe as mp

from model import keypoints, FRAMES


def make_tracker():
    return mp.solutions.holistic.Holistic(
        model_complexity=1, static_image_mode=False,
        min_detection_confidence=0.3, min_tracking_confidence=0.3)


def process_frame(tracker, bgr, flip=False):
    if flip:
        bgr = cv2.flip(bgr, 1)
    h, w = bgr.shape[:2]
    if w > 640:
        bgr = cv2.resize(bgr, (640, round(h * 640 / w)))
    results = tracker.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    has_pose = results.pose_landmarks is not None
    has_hand = results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
    for group in [results.left_hand_landmarks, results.right_hand_landmarks]:
        mp.solutions.drawing_utils.draw_landmarks(
            bgr, group, mp.solutions.holistic.HAND_CONNECTIONS)
    return keypoints(results), bgr, bool(has_pose and has_hand)


def extract_clip(path, start_seconds=0.0, flip=False):
    cap = cv2.VideoCapture(str(path))
    sequence, tracked, preview = [], 0, None
    try:
        if not cap.isOpened():
            raise ValueError("Cannot decode video. Try an MP4 recording.")
        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)
        with make_tracker() as tracker:
            # First 30 consecutive frames after the chosen start match training.
            while len(sequence) < FRAMES:
                ok, frame = cap.read()
                if not ok:
                    break
                vector, preview, valid = process_frame(tracker, frame, flip)
                sequence.append(vector)
                tracked += int(valid)
    finally:
        cap.release()
    if len(sequence) < 8:
        raise ValueError("Too few frames. Choose a start time earlier in the clip.")
    if tracked / len(sequence) < 0.5:
        raise ValueError("Tracking failed: show your upper body and hands in good lighting.")
    return sequence, preview, tracked / len(sequence)
