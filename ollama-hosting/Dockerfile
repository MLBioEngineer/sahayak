FROM ollama/ollama:latest

# Hugging Face Spaces exposes port 7860
ENV OLLAMA_HOST=0.0.0.0:7860
ENV OLLAMA_ORIGINS=*
ENV MODEL_NAME=llama3.2:3b

EXPOSE 7860

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
