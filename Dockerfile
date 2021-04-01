FROM python:3.7.3

WORKDIR '/app'

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["pytest", "--cov=ml_api", "ml_api/tests"]