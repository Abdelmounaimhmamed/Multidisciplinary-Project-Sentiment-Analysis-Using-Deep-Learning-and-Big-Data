FROM python:3.9 

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt



CMD ["python3", "Producer.py/Producer.py"]
