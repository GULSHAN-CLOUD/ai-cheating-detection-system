import time

class SuspicionTracker:
    def __init__(self):
        self.last_capture_time = 0
        self.cooldown_seconds = 5

        self.prev_direction = "forward"
        self.left_right_count = 0

    def can_capture(self):
        current_time = time.time()
        return (current_time - self.last_capture_time) >= self.cooldown_seconds

    def mark_captured(self):
        self.last_capture_time = time.time()

    def update_direction_count(self, direction):
        if direction in ["left", "right"]:
            if self.prev_direction != direction and self.prev_direction in ["left", "right", "forward"]:
                self.left_right_count += 1

        if direction == "no_face":
            self.left_right_count = 0

        self.prev_direction = direction

    def update(self, face_visible, person_count, phone_detected, suspicious_object_detected, direction):
        score = 0
        reasons = []
        capture_needed = False
        capture_reason = None

        self.update_direction_count(direction)

        if not face_visible:
            score += 2
            reasons.append("face missing")

        if person_count > 1:
            score += 5
            reasons.append("multiple persons")
            capture_needed = True
            capture_reason = "multiple_persons"

        if phone_detected:
            score += 5
            reasons.append("phone detected")
            capture_needed = True
            if capture_reason is None:
                capture_reason = "phone_detected"

        if suspicious_object_detected:
            score += 4
            reasons.append("suspicious object")
            capture_needed = True
            if capture_reason is None:
                capture_reason = "suspicious_object"

        if direction in ["left", "right", "down"]:
            score += 1
            reasons.append(f"looking {direction}")

        if self.left_right_count >= 3:
            reasons.append("looked left/right 3 times")
            capture_needed = True
            if capture_reason is None:
                capture_reason = "looked_left_right_3_times"

        return score, reasons, capture_needed, capture_reason, self.left_right_count