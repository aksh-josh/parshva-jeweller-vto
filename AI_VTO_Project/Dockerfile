# Use the official Python 3.11 image to match your local requirements
FROM python:3.11-slim

# Install system dependencies required by OpenCV, MediaPipe, and MySQL
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    pkg-config \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Point Docker to the official PyTorch CPU servers
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of your Python code and ML models into the container
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "final_app.py"]
