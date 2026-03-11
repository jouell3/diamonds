FROM python:3.11-slim

WORKDIR /diamonds

RUN pip install --upgrade pip setuptools wheel
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY api/ api/
RUN pip install .

CMD ["python", "api/api.py"]