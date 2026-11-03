# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY deployment_api.py .
COPY src/deployment_api.py src/airport_encodings.json src/iata_to_bts.json src/finalized_model.pkl ./src/

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "deployment_api:app", "--host", "0.0.0.0", "--port", "8000"]