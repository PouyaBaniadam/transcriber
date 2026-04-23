import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File

from logger_config import logger
from processor import process_audio_to_text

app = FastAPI(title="Professional STT API")


@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    req_id = str(uuid.uuid4())
    try:
        logger.info(f"ID: {req_id} | File: {file.filename}")
        content = await file.read()
        text = await process_audio_to_text(content)

        return {
            "id": req_id,
            "timestamp": datetime.utcnow().isoformat(),
            "text": text,
            "status": "success"
        }
    except Exception as e:
        return {"id": req_id, "error": str(e), "status": "failed"}


@app.get("/logs")
def get_logs():
    with open("logs/app.log", "r") as f:
        return {"logs": f.readlines()[-20:]}