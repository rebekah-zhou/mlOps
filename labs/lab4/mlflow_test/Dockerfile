# Use a Python base image
FROM python:3

# Set the working directory inside the container
WORKDIR /app

# Install mlflow
RUN pip install mlflow

# Expose port 5050 so it can be accessed from the host
EXPOSE 5050

# Run the MLflow server when the container starts on port 5050
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5050"]
