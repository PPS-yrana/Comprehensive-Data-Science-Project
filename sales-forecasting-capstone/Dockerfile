FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Dashboard on 8501; for the API use:
#   docker run -p 8000:8000 <image> uvicorn deployment.api:app --host 0.0.0.0 --port 8000
EXPOSE 8501
CMD ["streamlit", "run", "deployment/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
