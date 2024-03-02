.PHONY:
.ONESHELL:

# Show help message
help:
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# Install dependencies
run-install:
	apt-get -y install build-essential python3-dev ffmpeg
	pip install --upgrade setuptools wheel
	pip install --upgrade pip
	pip install faiss-gpu fairseq gradio ffmpeg ffmpeg-python praat-parselmouth pyworld numpy==1.23.5 numba==0.56.4 librosa==0.9.1
	pip install -r requirements.txt
	pip install --upgrade lxml
	apt-get update

# Run Applio
run-applio:
	python app.py --share

# Run Tensorboard
run-tensorboard:
	python core.py tensorboard
