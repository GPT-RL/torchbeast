#! /usr/bin/env bash
mkdir -p logs ~/.cache/GPT ~/.cache/huggingface
name=torchbeast_run
docker build -t $name .
docker run --rm -it --env-file .env --gpus "$1" \
	--shm-size 8G \
	-e HOST_MACHINE="$(hostname -s)" \
	-v "$(pwd)/logs:/root/logs" \
	-v "$(pwd)/dataset:/project/dataset" \
	-v "$HOME/.cache/GPT/:/root/.cache/GPT" \
	-v "$HOME/.cache/huggingface/:/root/.cache/huggingface" \
	$name "${@:2}"
