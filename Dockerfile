# syntax=docker/dockerfile:1

FROM python:3.9.18-bullseye

EXPOSE 7865

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

CMD ["python", "infer-web.py"]