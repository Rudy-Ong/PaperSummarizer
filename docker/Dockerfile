FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[notebook]"
COPY notebooks ./notebooks
COPY scripts ./scripts
ENV OLLAMA_HOST=http://host.docker.internal:11434
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password=", "--notebook-dir=/app"]
