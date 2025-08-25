FROM monkeyocr:latest
COPY . /app
WORKDIR /app

CMD ["python", "main.py"]