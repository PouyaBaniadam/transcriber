import io
import speech_recognition as sr
from pydub import AudioSegment
from logger_config import logger

recognizer = sr.Recognizer()


async def process_audio_to_text(audio_bytes: bytes, language: str = "fa-UR") -> str:
    try:
        logger.info(f"Processing audio with language: {language}")
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))

        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language=language)

        logger.info(f"Transcription successful ({language})")
        return text

    except sr.UnknownValueError:
        logger.warning(f"Speech was unintelligible in language: {language}")
        raise ValueError("Could not understand audio.")

    except Exception as e:
        logger.error(f"Error in processor: {str(e)}")
        raise e