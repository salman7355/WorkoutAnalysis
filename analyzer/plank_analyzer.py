import cv2
import mediapipe as mp
from typing import List, Tuple

from .pose_utils import calculate_angle, pick_visible_side, get_joint_positions

mp_pose = mp.solutions.pose

# Hip angle (shoulder-hip-ankle) must be at least this close to straight
# (180 = perfectly straight) to count as aligned. Same idea as the push-up
# back-sag check, but here it's evaluated continuously over time, not
# per-rep, since a plank has no reps.
ALIGNMENT_ANGLE_THRESHOLD = 160

MIN_ISSUE_DURATION_SEC = 1.5

ISSUE_TIP = (
    "Keep your body in a straight line from shoulders to ankles - "
    "avoid letting your hips sag or pike up."
)


def analyze_plank_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_index = 0
    frame_records: List[Tuple[float, bool]] = []  

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        timestamp = frame_index / fps
        frame_index += 1

        if not results.pose_landmarks:
            continue

        landmarks = results.pose_landmarks.landmark
        side = pick_visible_side(landmarks, mp_pose)
        joints = get_joint_positions(landmarks, mp_pose, side)

        hip_angle = calculate_angle(joints["shoulder"], joints["hip"], joints["ankle"])
        is_aligned = hip_angle >= ALIGNMENT_ANGLE_THRESHOLD

        frame_records.append((timestamp, is_aligned))

    cap.release()
    pose.close()

    if not frame_records:
        raise ValueError("No pose detected in video")

    total_hold_seconds = frame_records[-1][0] - frame_records[0][0]
    aligned_frame_count = sum(1 for _, aligned in frame_records if aligned)
    aligned_seconds = aligned_frame_count / fps
    misaligned_seconds = max(0.0, total_hold_seconds - aligned_seconds)

    form_score_percent = (
        round((aligned_seconds / total_hold_seconds) * 100, 1)
        if total_hold_seconds > 0
        else 0.0
    )

    return {
        "exercise": "plank",
        "totalHoldSeconds": round(total_hold_seconds, 1),
        "alignedSeconds": round(aligned_seconds, 1),
        "misalignedSeconds": round(misaligned_seconds, 1),
        "formScorePercent": form_score_percent,
        "issues": _extract_issue_segments(frame_records),
    }


def _extract_issue_segments(frame_records: List[Tuple[float, bool]]) -> list:
    """Merges consecutive misaligned frames into segments, dropping ones
    too short to be real bad form rather than a single noisy frame."""
    segments = []
    segment_start = None

    for timestamp, aligned in frame_records:
        if not aligned and segment_start is None:
            segment_start = timestamp
        elif aligned and segment_start is not None:
            _maybe_add_segment(segments, segment_start, timestamp)
            segment_start = None

    if segment_start is not None:
        _maybe_add_segment(segments, segment_start, frame_records[-1][0])

    return segments


def _maybe_add_segment(segments: list, start: float, end: float) -> None:
    if end - start >= MIN_ISSUE_DURATION_SEC:
        segments.append(
            {
                "startSec": round(start, 1),
                "endSec": round(end, 1),
                "issue": "hips_not_aligned",
                "tip": ISSUE_TIP,
            }
        )