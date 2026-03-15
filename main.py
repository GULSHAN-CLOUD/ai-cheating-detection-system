import cv2
import os
import time
from detector import detect_objects
from face_tracker import get_face_landmarks, get_head_direction
from rules import SuspicionTracker

os.makedirs("captures", exist_ok=True)

cap = cv2.VideoCapture(0)
tracker = SuspicionTracker()

SUSPICIOUS_OBJECTS = ["book", "cell phone", "laptop", "tablet"]

def save_capture(frame, reason_text):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"captures/{timestamp}_{reason_text}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    detections = detect_objects(frame)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["conf"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    landmarks = get_face_landmarks(frame)

    if landmarks is not None:
        h, w, _ = frame.shape
        for lm in landmarks[:50]:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    face_visible = landmarks is not None
    face_text = "FACE VISIBLE" if face_visible else "FACE NOT VISIBLE"
    face_color = (0, 255, 0) if face_visible else (0, 0, 255)

    cv2.putText(frame, face_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, face_color, 2)

    person_count = sum(1 for d in detections if d["label"] == "person")
    cv2.putText(frame, f"Persons: {person_count}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    phone_detected = any(d["label"] == "cell phone" for d in detections)
    phone_text = "PHONE DETECTED" if phone_detected else "NO PHONE"
    phone_color = (0, 0, 255) if phone_detected else (0, 255, 0)

    cv2.putText(frame, phone_text, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, phone_color, 2)

    suspicious_object_detected = any(
        d["label"] in SUSPICIOUS_OBJECTS for d in detections if d["label"] != "person"
    )

    suspicious_text = "SUSPICIOUS OBJECT" if suspicious_object_detected else "NO SUSPICIOUS OBJECT"
    suspicious_color = (0, 0, 255) if suspicious_object_detected else (0, 255, 0)

    cv2.putText(frame, suspicious_text, (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, suspicious_color, 2)

    direction = get_head_direction(landmarks)
    cv2.putText(frame, f"Head: {direction}", (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    score, reasons, capture_needed, capture_reason, left_right_count = tracker.update(
        face_visible,
        person_count,
        phone_detected,
        suspicious_object_detected,
        direction
    )

    cv2.putText(frame, f"Score: {score}", (20, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.putText(frame, f"Left/Right Count: {left_right_count}", (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if score >= 8:
        status = "HIGH RISK"
        status_color = (0, 0, 255)
    elif score >= 4:
        status = "SUSPICIOUS"
        status_color = (0, 165, 255)
    else:
        status = "NORMAL"
        status_color = (0, 255, 0)

    cv2.putText(frame, f"Status: {status}", (20, 275),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)

    y = 310
    for r in reasons[:3]:
        cv2.putText(frame, r, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 30

    # Strict 5-second cooldown
    if capture_needed:
        if tracker.can_capture():
            save_capture(frame, capture_reason)
            tracker.mark_captured()

            if left_right_count >= 3:
                tracker.left_right_count = 0

    cv2.imshow("Cheating Detection System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()