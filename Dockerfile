FROM python:3.8-slim-buster

# 1. Install AWS CLI and immediately clean up the apt cache to save space
RUN apt update -y && \
    apt install awscli -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. CACHING: Copy ONLY the files needed for installation first
COPY setup.py requirements.txt README.md ./
COPY src/ ./src/

# 3. INSTALLATION: Chain all your pip commands into a SINGLE run command.
# This fixes your dependency conflict without bloating the image size.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade accelerate && \
    pip uninstall -y transformers accelerate && \
    pip install --no-cache-dir transformers accelerate

# 4. Copy the rest of your application code
COPY . /app

# 5. Boot the app
CMD ["python3", "app.py"]