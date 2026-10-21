.PHONY:
.ONESHELL:

# Show help message
help:
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# Install dependencies
run-install:
	apt-get update
	apt-get -y install build-essential python3-dev ffmpeg
	uv sync --locked

# Run Applio
run-applio:
	uv run app.py --share

# Run Tensorboard
run-tensorboard:
	uv run core.py tensorboard

# Regenerate requirements*.txt from uv.lock (used by the Windows updater and plugins)
requirements:
	uv export --frozen --no-hashes --no-emit-project -o requirements.txt
	uv export --frozen --no-default-groups --no-hashes --no-emit-project -o requirements-engine.txt
