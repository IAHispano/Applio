# syntax=docker/dockerfile:1

FROM python:3.10-bullseye

EXPOSE 6969

WORKDIR /app

RUN apt update && apt install -y -qq ffmpeg aria2 && apt clean

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

VOLUME [ "/app/logs/weights", "/app/opt" ]

ENTRYPOINT [ "python3" ]

CMD ["app.py"]
