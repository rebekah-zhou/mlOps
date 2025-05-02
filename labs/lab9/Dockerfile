FROM python:3.11-slim

RUN pip install --upgrade pip

WORKDIR /app
COPY ["requirements.txt", "./"]
RUN pip install -r requirements.txt

COPY ["reddit_model_pipeline.joblib", "redditApp.py", "./"]

EXPOSE 8000

ENTRYPOINT ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]
