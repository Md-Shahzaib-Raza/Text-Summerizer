FROM python:3.8-slim

WORKDIR /app

COPY setup.py requirements.txt README.md ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "app.py"]
