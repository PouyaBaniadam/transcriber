import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


MODEL_PATH = "/home/pouya/Downloads/whisper"
AUDIO_FILE = "voice/1.mp3"

if not os.path.exists(MODEL_PATH):
    print(f"!!! ERROR: Path {MODEL_PATH} not found.")
    exit(1)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

try:
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True
    ).to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    print(f"--- Starting TRANSLATION Task on: {AUDIO_FILE} ---")

    result = pipe(
        AUDIO_FILE,
        generate_kwargs={"task": "translate"},
        chunk_length_s=5,
        batch_size=8,
        return_timestamps=False
    )

    print("\n--- TRANSLATION RESULT (Full Text) ---\n")
    print(result["text"])

    print("\n--- TIMESTAMPS (Segment by Segment) ---\n")
    if "chunks" in result:
        for chunk in result["chunks"]:
            start_time = chunk["timestamp"][0]
            end_time = chunk["timestamp"][1]
            text = chunk["text"]

            end_str = f"{end_time:.2f}" if end_time is not None else "End"

            print(f"[{start_time:.2f}s -> {end_str}s] : {text}")

except Exception as e:
    print(f"An error occurred: {e}")