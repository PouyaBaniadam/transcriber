import uuid
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query

from logger_config import logger
from processor import process_audio_to_text

app = FastAPI(title="Professional STT API")


@app.post("/transcribe/")
async def transcribe(
        file: UploadFile = File(...),
        lang: str = Query("fa-IR", description="Language code (e.g., 'en-US', 'fa-IR', 'tr-TR')")
):
    req_id = str(uuid.uuid4())

    try:
        logger.info(f"ID: {req_id} | File: {file.filename} | Target Lang: {lang}")

        content = await file.read()

        text = await process_audio_to_text(content, language=lang)

        return {
            "id": req_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "filename": file.filename,
            "language": lang,
            "transcription": text,
            "status": "success"
        }

    except ValueError as ve:
        return {"id": req_id, "error": str(ve), "status": "failed"}
    except Exception as e:
        logger.error(f"Server Error on request {req_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/languages")
def get_supported_languages():
    return {
        "common_codes": {
            "English (US)": "en-US",
            "Persian (IR)": "fa-IR",
            "Turkish (TR)": "tr-TR",
            "Arabic (AR)": "ar-SA",
            "French (FR)": "fr-FR"
        },
        "info": "You can use any standard ISO language code."
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)