# AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10-arm64
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /var/task

# -----------------------------
# 1) Install Python dependencies
# -----------------------------
# Expect a file at: lambda/requirements.txt
COPY lambda/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 2) Copy application code
# -----------------------------
# Lambda handler and any helper modules
COPY lambda/*.py ./

# ETL / models used by the handler
COPY etl/ ./etl/
COPY models/ ./models/
# (Optional) if you reference SQL files or others at runtime, copy them too:
# COPY sql/ ./sql/

# Default region (can be overridden by Lambda environment variables)
ENV AWS_REGION=ap-northeast-2

# -----------------------------
# 3) Set the Lambda handler
# -----------------------------
# File: lambda_reallocation.py, Function: handler
CMD ["lambda_reallocation.handler"]