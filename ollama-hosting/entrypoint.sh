#!/bin/sh
set -e

echo "Starting Ollama server on ${OLLAMA_HOST}..."
ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama to become ready..."
until curl -s http://127.0.0.1:7860/api/tags > /dev/null; do
    sleep 2
done

echo "Pulling quantized model: ${MODEL_NAME}..."
ollama pull "${MODEL_NAME}"

echo "Model ${MODEL_NAME} is ready to serve inference requests!"
wait $SERVER_PID
