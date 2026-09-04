#!/bin/sh

# $1 is the torch extra to install: cpu or cu128. They are declared as
# conflicting extras in pyproject.toml, each pinned to its own PyTorch index.
extra="${1:-cpu}"

sudo chown -R $(whoami):$(whoami) /home/$(whoami)/app/.venv
uv sync --extra "${extra}"
