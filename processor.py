import io
import speech_recognition as sr
from pydub import AudioSegment
from logger_config import logger

recognizer = sr.Recognizer()


async def process_audio_to_text(audio_bytes: bytes) -> str:
    try:
        logger.info("Processing audio...")
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))

        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language="en-US")
        logger.info(f"Success: {text[:30]}...")
        return text
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise e