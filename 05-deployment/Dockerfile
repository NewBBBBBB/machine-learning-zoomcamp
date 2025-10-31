FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

COPY pyproject.toml .

RUN pip install uv && \
    uv pip compile pyproject.toml

COPY pipeline_v1.bin .
COPY predict.py .

EXPOSE 8000

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]