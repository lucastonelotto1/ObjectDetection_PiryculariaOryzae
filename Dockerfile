# Use official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the current directory contents into the container at /code
COPY . .

# Create a writable directory for cache/temporary files if needed
RUN mkdir -p /tmp/cache && chmod 777 /tmp/cache

# Define environment variable
ENV HOME=/tmp/cache

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run uvicorn server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
