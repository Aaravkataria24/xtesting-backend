FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and application code
COPY finetuned_twitter_roberta_multi.pt .
COPY regressor_model.pkl .
COPY main.py .

# Expose port for the application
EXPOSE 8080

# Command to run the application
CMD ["python", "main.py"]