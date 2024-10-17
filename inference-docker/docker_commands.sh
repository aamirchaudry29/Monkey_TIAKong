docker build . -t tiakong -f inference-docker/dockerfile
docker run -it --gpus all --network none tiakong