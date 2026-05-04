import io
import os
import torch
import speech_recognition as sr
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from logger_config import logger

# --- Local Whisper Configuration ---
MODEL_PATH = "/home/pouya/Downloads/openai⁄whisper-large-v3-turbo"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Initialize Global Whisper Pipeline (Loaded once)
logger.info(f"Loading Local Whisper Model from {MODEL_PATH}...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True
    ).to(device)

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    logger.info("Local Whisper Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load local Whisper model: {e}")
    whisper_pipe = None

# Initialize Google Recognizer
recognizer = sr.Recognizer()


async def process_audio_to_text(file_path: str, language: str = "fa-IR", is_local: bool = True) -> str:
    """
    Switching logic between Local Whisper and Google Speech Recognition
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if is_local:
        if whisper_pipe is None:
            raise ValueError("Local Whisper model is not loaded correctly.")

        logger.info(f"Processing LOCALLY: {file_path}")
        # Whisper uses short codes (fa, en, etc.)
        short_lang = language.split("-")[0]

        result = whisper_pipe(
            file_path,
            generate_kwargs={"language": short_lang, "task": "transcribe"},
            chunk_length_s=30,
            batch_size=8
        )
        return result["text"]

    else:
        logger.info(f"Processing via GOOGLE: {file_path}")
        # Read file and convert to WAV for speech_recognition
        audio_segment = AudioSegment.from_file(file_path)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)

        return recognizer.recognize_google(audio_data, language=language)