set shell := ["bash", "-euo", "pipefail", "-c"]

# Set up dependencies (default env/kernel)
setup:
    @echo "Syncing project dependencies (including dev) with uv..."
    uv sync --dev
    @echo "Done. Use 'just lab' to launch JupyterLab."

# Launch JupyterLab in the current environment
lab:
    uv run jupyter-lab
