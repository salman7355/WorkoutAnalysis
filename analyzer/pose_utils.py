import numpy as np


def calculate_angle(a, b, c):
    """
    Returns the angle (in degrees) at point b, formed by points a-b-c.
    Each point is (x, y) in normalized image coordinates (0-1).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


def pick_visible_side(landmarks, mp_pose):
    """
    Side-view videos usually only clearly show one side of the body.
    Pick left or right based on whichever has higher average visibility
    across the joints we actually care about.
    """
    left_indices = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
    ]
    right_indices = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]

    left_visibility = np.mean([landmarks[i.value].visibility for i in left_indices])
    right_visibility = np.mean([landmarks[i.value].visibility for i in right_indices])

    return "left" if left_visibility >= right_visibility else "right"


def get_joint_positions(landmarks, mp_pose, side):
    prefix = side.upper()
    lm = mp_pose.PoseLandmark

    def pos(name):
        point = landmarks[getattr(lm, f"{prefix}_{name}").value]
        return (point.x, point.y)

    return {
        "shoulder": pos("SHOULDER"),
        "elbow": pos("ELBOW"),
        "wrist": pos("WRIST"),
        "hip": pos("HIP"),
        "knee": pos("KNEE"),
        "ankle": pos("ANKLE"),
    }