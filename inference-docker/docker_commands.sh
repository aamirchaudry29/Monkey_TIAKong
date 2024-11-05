docker build . -t tiakong -f inference-docker/dockerfile
docker run -it --gpus all --network none tiakong

tar -czvf algorithmmodel.tar.gz -C weights/ .
tar -xvzf algorithmmodel.tar.gz