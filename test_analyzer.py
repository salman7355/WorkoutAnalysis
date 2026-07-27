"""
Quick local test for the analyzers, bypassing FastAPI/Supabase entirely.

Usage:
    python test_analyzer.py path/to/video.mp4 push_up
    python test_analyzer.py path/to/video.mp4 squat
"""
import sys
import json
from analyzer.pushup_analyzer import analyze_pushup_video
from analyzer.squat_analyzer import analyze_squat_video

ANALYZERS = {
    "push_up": analyze_pushup_video,
    "squat": analyze_squat_video,
}

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python test_analyzer.py path/to/video.mp4 [push_up|squat]")
        sys.exit(1)

    video_path = sys.argv[1]
    exercise_type = sys.argv[2] if len(sys.argv) == 3 else "push_up"

    analyzer_fn = ANALYZERS.get(exercise_type)
    if analyzer_fn is None:
        print(f"Unknown exercise type: {exercise_type}. Supported: {list(ANALYZERS.keys())}")
        sys.exit(1)

    print(f"Analyzing {video_path} as {exercise_type}...")

    result = analyzer_fn(video_path)
    print(json.dumps(result, indent=2))