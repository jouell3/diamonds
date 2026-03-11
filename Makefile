

.PHONY: all
all: main.py sandbank.ipynb

test_api:
	curl -X 'GET' \"http://127.0.0.1:8000\"

test_prediction:
	curl -X 'GET' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"carat": 0.23, "cut": "Ideal", "color": "E", "clarity": "SI2", "depth": 61.5, "table": 55.0, "x": 3.95, "y": 3.98, "z": 2.43}'

build: ## Build Docker image locally
	docker build -t ${IMAGE} .

run: build ## Build and run container locally
	docker run -p ${PORT}:${PORT} -e PORT=${PORT} ${IMAGE} 

build_gcp: ## Build image for GCP (Linux/amd64 platform)
	@echo "Building the image for GCP..."
	docker buildx build --platform linux/amd64 -t ${LOCATION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_REPOSITORY}/${IMAGE} . --push

push_gcp: build_gcp ## Build and push image to Artifact Registry
	docker push ${LOCATION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_REPOSITORY}/${IMAGE}