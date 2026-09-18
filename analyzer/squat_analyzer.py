import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .pose_utils import calculate_angle, pick_visible_side, get_joint_positions

mp_pose = mp.solutions.pose

STAND_THRESHOLD = 160
SQUAT_THRESHOLD = 120
REQUIRED_DEPTH_ANGLE = 100
LOCKOUT_ANGLE = 160
FORWARD_LEAN_MIN_ANGLE = 40

SMOOTHING_WINDOW = 3


@dataclass
class RepMetrics:
    min_knee_angle: float = 180.0
    min_hip_angle: float = 180.0
    lockout_angle: Optional[float] = None
    start_time: float = 0.0  # timestamp (sec) when the descent for this rep began


@dataclass
class RepResult:
    rep_index: int
    is_valid: bool
    issue: Optional[str] = None
    tip: Optional[str] = None
    start_sec: float = 0.0
    end_sec: float = 0.0


ISSUE_TIPS = {
    "insufficient_depth": "Squat deeper - aim to get your hip crease at least level with your knees.",
    "leaning_too_far_forward": "Keep your chest up and torso more upright as you descend.",
    "incomplete_lockout": "Stand all the way up and squeeze your glutes at the top of each rep.",
}


def _moving_average(values, window):
    if len(values) < window:
        return values[-1]
    return float(np.mean(values[-window:]))


def _evaluate_rep(rep_index: int, metrics: RepMetrics) -> RepResult:
    if metrics.min_knee_angle > REQUIRED_DEPTH_ANGLE:
        return RepResult(rep_index, False, "insufficient_depth", ISSUE_TIPS["insufficient_depth"])

    if metrics.min_hip_angle < FORWARD_LEAN_MIN_ANGLE:
        return RepResult(rep_index, False, "leaning_too_far_forward", ISSUE_TIPS["leaning_too_far_forward"])

    if metrics.lockout_angle is not None and metrics.lockout_angle < LOCKOUT_ANGLE:
        return RepResult(rep_index, False, "incomplete_lockout", ISSUE_TIPS["incomplete_lockout"])

    return RepResult(rep_index, True)


def analyze_squat_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    knee_angle_history = []
    state = "up"
    current_rep_metrics: Optional[RepMetrics] = None
    completed_reps: list[RepResult] = []
    pending_lockout_rep: Optional[RepResult] = None
    frame_index = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        timestamp = frame_index / fps
        frame_index += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            continue

        landmarks = results.pose_landmarks.landmark
        side = pick_visible_side(landmarks, mp_pose)
        joints = get_joint_positions(landmarks, mp_pose, side)

        knee_angle = calculate_angle(joints["hip"], joints["knee"], joints["ankle"])
        hip_angle = calculate_angle(joints["shoulder"], joints["hip"], joints["knee"])

        knee_angle_history.append(knee_angle)
        smoothed_knee = _moving_average(knee_angle_history, SMOOTHING_WINDOW)

        if state == "up" and pending_lockout_rep is not None:
            if current_rep_metrics is None:
                current_rep_metrics = RepMetrics()
            current_rep_metrics.lockout_angle = max(
                current_rep_metrics.lockout_angle or 0.0, smoothed_knee
            )

        if state == "up" and smoothed_knee < SQUAT_THRESHOLD:
            if pending_lockout_rep is not None and current_rep_metrics is not None:
                if (
                    pending_lockout_rep.is_valid
                    and current_rep_metrics.lockout_angle is not None
                    and current_rep_metrics.lockout_angle < LOCKOUT_ANGLE
                ):
                    pending_lockout_rep.is_valid = False
                    pending_lockout_rep.issue = "incomplete_lockout"
                    pending_lockout_rep.tip = ISSUE_TIPS["incomplete_lockout"]
                pending_lockout_rep = None

            state = "down"
            current_rep_metrics = RepMetrics()
            current_rep_metrics.min_knee_angle = smoothed_knee
            current_rep_metrics.min_hip_angle = hip_angle
            current_rep_metrics.start_time = timestamp

        elif state == "down":
            current_rep_metrics.min_knee_angle = min(
                current_rep_metrics.min_knee_angle, smoothed_knee
            )
            current_rep_metrics.min_hip_angle = min(
                current_rep_metrics.min_hip_angle, hip_angle
            )

            if smoothed_knee > STAND_THRESHOLD:
                rep_index = len(completed_reps)
                rep_result = _evaluate_rep(rep_index, current_rep_metrics)
                rep_result.start_sec = round(current_rep_metrics.start_time, 2)
                rep_result.end_sec = round(timestamp, 2)
                completed_reps.append(rep_result)
                pending_lockout_rep = rep_result
                state = "up"
                current_rep_metrics = RepMetrics()

    cap.release()
    pose.close()

    valid_reps = sum(1 for r in completed_reps if r.is_valid)

    return {
        "type": "reps",
        "exercise": "squat",
        "totalReps": len(completed_reps),
        "validReps": valid_reps,
        "invalidReps": len(completed_reps) - valid_reps,
        "feedback": [
            {
                "repIndex": r.rep_index,
                "isValid": r.is_valid,
                "issue": r.issue,
                "tip": r.tip,
                "startSec": r.start_sec,
                "endSec": r.end_sec,
            }
            for r in completed_reps
        ],
    }