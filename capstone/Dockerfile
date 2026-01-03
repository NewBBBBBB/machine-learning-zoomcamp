# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app


# Copy dependency file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY predict.py .
COPY sentiment_model.pkl .

# Expose API port
EXPOSE 9696

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
