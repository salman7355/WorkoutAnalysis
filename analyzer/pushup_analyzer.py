import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .pose_utils import calculate_angle, pick_visible_side, get_joint_positions

mp_pose = mp.solutions.pose


UP_THRESHOLD = 155          # elbow angle above this = considered "up" / starting a rep
DOWN_THRESHOLD = 100        # elbow angle below this = considered "down" / bottom of rep
REQUIRED_DEPTH_ANGLE = 100  # must reach at or below this to count as sufficient depth
LOCKOUT_ANGLE = 155         # must reach at or above this at top to count as full lockout
BACK_STRAIGHT_TOLERANCE = 20  # hip angle must stay within 180 +/- this to be "straight"
ELBOW_FLARE_MAX = 80        # torso-to-upper-arm angle above this = excessive flare

# Smoothing window for landmark jitter
SMOOTHING_WINDOW = 3


@dataclass
class RepMetrics:
    min_elbow_angle: float = 180.0
    max_back_deviation: float = 0.0  # abs deviation from 180 at hip
    max_flare_angle: float = 0.0
    lockout_angle: Optional[float] = None  # filled in after the "up" phase completes


@dataclass
class RepResult:
    rep_index: int
    is_valid: bool
    issue: Optional[str] = None
    tip: Optional[str] = None


ISSUE_TIPS = {
    "insufficient_depth": "Lower your chest closer to the ground for a full range of motion.",
    "back_not_straight": "Keep your core tight and your body in a straight line from shoulders to ankles.",
    "elbow_flare": "Keep your elbows closer to your body, around a 45 degree angle, to protect your shoulders.",
    "incomplete_lockout": "Fully extend your arms at the top of each rep.",
}


def _moving_average(values, window):
    if len(values) < window:
        return values[-1]
    return float(np.mean(values[-window:]))


def _evaluate_rep(rep_index: int, metrics: RepMetrics) -> RepResult:
    """Check a completed rep's metrics against thresholds, in priority order."""
    if metrics.min_elbow_angle > REQUIRED_DEPTH_ANGLE:
        return RepResult(rep_index, False, "insufficient_depth", ISSUE_TIPS["insufficient_depth"])

    if metrics.max_back_deviation > BACK_STRAIGHT_TOLERANCE:
        return RepResult(rep_index, False, "back_not_straight", ISSUE_TIPS["back_not_straight"])

    if metrics.max_flare_angle > ELBOW_FLARE_MAX:
        return RepResult(rep_index, False, "elbow_flare", ISSUE_TIPS["elbow_flare"])

    if metrics.lockout_angle is not None and metrics.lockout_angle < LOCKOUT_ANGLE:
        return RepResult(rep_index, False, "incomplete_lockout", ISSUE_TIPS["incomplete_lockout"])

    return RepResult(rep_index, True)


def analyze_pushup_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    elbow_angle_history = []
    state = "up"  # "up" or "down"
    current_rep_metrics: Optional[RepMetrics] = None
    completed_reps: list[RepResult] = []
    pending_lockout_rep: Optional[RepResult] = None  # rep awaiting its lockout check

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            continue

        landmarks = results.pose_landmarks.landmark
        side = pick_visible_side(landmarks, mp_pose)
        joints = get_joint_positions(landmarks, mp_pose, side)

        elbow_angle = calculate_angle(joints["shoulder"], joints["elbow"], joints["wrist"])
        hip_angle = calculate_angle(joints["shoulder"], joints["hip"], joints["ankle"])
        flare_angle = calculate_angle(joints["hip"], joints["shoulder"], joints["elbow"])

        elbow_angle_history.append(elbow_angle)
        smoothed_elbow = _moving_average(elbow_angle_history, SMOOTHING_WINDOW)

        back_deviation = abs(180.0 - hip_angle)

        # If we're in the "up" phase, keep tracking peak elbow angle for lockout
        # of whichever rep just completed (if any).
        if state == "up" and pending_lockout_rep is not None:
            if current_rep_metrics is None:
                current_rep_metrics = RepMetrics()
            current_rep_metrics.lockout_angle = max(
                current_rep_metrics.lockout_angle or 0.0, smoothed_elbow
            )

        if state == "up" and smoothed_elbow < DOWN_THRESHOLD:
            # Descent started - finalize the previous rep's lockout check, if any
            if pending_lockout_rep is not None and current_rep_metrics is not None:
                pending_lockout_rep.tip = pending_lockout_rep.tip  # no-op, kept for clarity
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
            current_rep_metrics.min_elbow_angle = smoothed_elbow
            current_rep_metrics.max_back_deviation = back_deviation
            current_rep_metrics.max_flare_angle = flare_angle

        elif state == "down":
            current_rep_metrics.min_elbow_angle = min(
                current_rep_metrics.min_elbow_angle, smoothed_elbow
            )
            current_rep_metrics.max_back_deviation = max(
                current_rep_metrics.max_back_deviation, back_deviation
            )
            current_rep_metrics.max_flare_angle = max(
                current_rep_metrics.max_flare_angle, flare_angle
            )

            if smoothed_elbow > UP_THRESHOLD:
                # Ascent completed - count the rep, defer lockout check to the up phase
                rep_index = len(completed_reps)
                rep_result = _evaluate_rep(rep_index, current_rep_metrics)
                completed_reps.append(rep_result)
                pending_lockout_rep = rep_result
                state = "up"
                current_rep_metrics = RepMetrics()  # reset for lockout tracking

    cap.release()
    pose.close()

    valid_reps = sum(1 for r in completed_reps if r.is_valid)

    return {
        "exercise": "push_up",
        "totalReps": len(completed_reps),
        "validReps": valid_reps,
        "invalidReps": len(completed_reps) - valid_reps,
        "feedback": [
            {
                "repIndex": r.rep_index,
                "isValid": r.is_valid,
                "issue": r.issue,
                "tip": r.tip,
            }
            for r in completed_reps
        ],
    }