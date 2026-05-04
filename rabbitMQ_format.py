import os
import json
import uuid
import pika
import asyncio
import threading
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from logger_config import logger
from processor import process_audio_to_text

# --- RabbitMQ Configuration ---
RABBITMQ_HOST = "192.168.200.165"
RABBITMQ_PORT = 18011
QUEUE_INPUT = "transcription_input_queue"
QUEUE_OUTPUT = "transcription_output_queue"


def process_message(ch, method, properties, body):
    try:
        # 1. Parse Input Arguments
        data = json.loads(body.decode())
        file_path = data.get("file_path")
        req_id = data.get("uuid", str(uuid.uuid4()))
        lang = data.get("lang", "fa-IR")
        is_local = data.get("is_local", True)  # Default to local if not provided

        logger.info(f"Job Received | ID: {req_id} | Local: {is_local} | Lang: {lang}")

        # 2. Process
        # We pass the file_path and is_local boolean to the processor
        text = asyncio.run(process_audio_to_text(
            file_path=file_path,
            language=lang,
            is_local=is_local
        ))

        # 3. Prepare Success Output
        result = {
            "id": req_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "file_path": file_path,
            "language": lang,
            "is_local": is_local,
            "transcription": text,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error on request {req_id}: {str(e)}")
        result = {
            "id": req_id,
            "error": str(e),
            "status": "failed"
        }

    finally:
        # 4. Publish result
        ch.basic_publish(
            exchange="",
            routing_key=QUEUE_OUTPUT,
            body=json.dumps(result)
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info(f"Job Finished | ID: {req_id}")


def rabbitmq_worker():
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, heartbeat=0)
        )
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_INPUT)
        channel.queue_declare(queue=QUEUE_OUTPUT)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=QUEUE_INPUT, on_message_callback=process_message)

        logger.info(f" [*] RabbitMQ Worker connected. Listening on {QUEUE_INPUT}")
        channel.start_consuming()
    except Exception as e:
        logger.error(f"RabbitMQ Worker crashed: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_thread = threading.Thread(target=rabbitmq_worker, daemon=True)
    worker_thread.start()
    yield


app = FastAPI(title="STT Worker Service", lifespan=lifespan)

if __name__ == "__main__":
    uvicorn.run("rabbitMQ_format:app", host="0.0.0.0", port=8000)