# Dockerfile

# Change this line:
FROM python:3.11-slim
# (or python:3.10-slim if you want to try the smaller image first)

# Rest of the Dockerfile remains the same:
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.app:app"]