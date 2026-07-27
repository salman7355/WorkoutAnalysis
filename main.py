import os
import tempfile
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from analyzer.pushup_analyzer import analyze_pushup_video
from analyzer.squat_analyzer import analyze_squat_video

app = FastAPI()

ANALYZERS = {
    "push_up": analyze_pushup_video,
    "squat": analyze_squat_video,
}


class AnalyzeRequest(BaseModel):
    sessionId: str
    videoUrl: str  
    exerciseType: str = "push_up"


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    analyzer_fn = ANALYZERS.get(req.exerciseType)
    if analyzer_fn is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported exercise type: {req.exerciseType}. Supported: {list(ANALYZERS.keys())}",
        )

    tmp_path = None
    try:
        # Download the video to a temp file (deleted after analysis)
        response = requests.get(req.videoUrl, stream=True, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name

        result = analyzer_fn(tmp_path)
        result["sessionId"] = req.sessionId
        return result

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to download video: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok"}