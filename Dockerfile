FROM python:3.11-alpine

WORKDIR /app

COPY Project/frontend/ /app/

EXPOSE 3000

CMD ["python", "-m", "http.server", "3000", "--directory", "/app"]
