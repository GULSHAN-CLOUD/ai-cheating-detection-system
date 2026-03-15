import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_face_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    face = results.multi_face_landmarks[0]
    return face.landmark

def get_head_direction(landmarks):
    if landmarks is None:
        return "no_face"

    nose = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    chin = landmarks[152]
    forehead = landmarks[10]

    left_dist = abs(nose.x - left_cheek.x)
    right_dist = abs(right_cheek.x - nose.x)
    up_dist = abs(nose.y - forehead.y)
    down_dist = abs(chin.y - nose.y)

    if left_dist > right_dist + 0.03:
        return "right"
    elif right_dist > left_dist + 0.03:
        return "left"
    elif down_dist > up_dist + 0.08:
        return "down"
    else:
        return "forward"