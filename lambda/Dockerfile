FROM public.ecr.aws/lambda/python:3.10-arm64
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /var/task

COPY lambda/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY lambda/*.py ./

COPY etl/ ./etl/
COPY models/ ./models/
ENV AWS_REGION=ap-northeast-2


CMD ["lambda_reallocation.handler"]