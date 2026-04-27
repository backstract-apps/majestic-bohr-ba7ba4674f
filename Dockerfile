# Use the official Backstract Python slim image for the build
FROM kathyrussells/backstract-python-app-30-01-26:slim3.11

WORKDIR /usr/src/app

# To use env locally, rename .env.example to .env
# COPY .env to .env
# COPY .env.example .env

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*
    
# Copy and install dependencies
COPY ./requirements.txt .
RUN uv pip install -r requirements.txt --system

# Copy the project files
COPY . .

# Run the application
#CMD ["gunicorn", "main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
CMD ["gunicorn", "main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7070"]
