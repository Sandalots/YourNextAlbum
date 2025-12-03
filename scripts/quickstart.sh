#!/bin/zsh
set -e

# YourNextAlbum Quickstart script (runs from repo root even when called from scripts/)

# Resolve repository root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Select preferred Python (3.11, then 3.10, then 3.9)
PYTHON_BIN=""
for candidate in python3.11 python3.10 python3.9; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

# Fallback to python3 only if it matches an allowed minor version
if [ -z "$PYTHON_BIN" ] && command -v python3 >/dev/null 2>&1; then
  ver_minor="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
  case "$ver_minor" in
    3.11|3.10|3.9)
      PYTHON_BIN="python3"
      ;;
    *)
      PYTHON_BIN=""
      ;;
  esac
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "Error: Python 3.11, 3.10, or 3.9 not found in PATH." >&2
  echo "Hint: Install with pyenv, then set locally, e.g.:" >&2
  echo "  brew install pyenv" >&2
  echo "  pyenv install 3.11.9" >&2
  echo "  pyenv local 3.11.9" >&2
  echo "  python -V  # should show 3.11.x" >&2
  exit 1
fi

echo "Using Python interpreter: $PYTHON_BIN ($($PYTHON_BIN -V 2>&1))"

# 0. Create venv if neither .venv nor venv exists at repo root
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
  echo "Creating Python virtual environment (.venv) at $REPO_ROOT..."
  "$PYTHON_BIN" -m venv .venv
fi

# 1. Activate venv (prefer .venv, fallback to venv)
if [ -d ".venv" ]; then
  echo "Activating .venv..."
  source .venv/bin/activate
elif [ -d "venv" ]; then
  echo "Activating venv..."
  source venv/bin/activate
else
  echo "No virtual environment found and creation failed." >&2
  exit 1
fi

# 2. Install dependencies for YourNextAlbum
if [ -f requirements.txt ]; then
  echo "Installing dependencies from requirements.txt..."
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "requirements.txt not found at $REPO_ROOT" >&2
  exit 1
fi

# 3. Create album reviews dataset from the scraped data
echo "Running dataset creation..."
python album_reviews_dataset_creator.py

# 4. Analyze sentiment of album reviews
echo "Running sentiment analysis..."
python album_reviews_text_sentiment_analyser.py

# 5. Build album recommender models
echo "Building recommender models..."
python album_recommender_model.py

echo "Quickstart complete: env ready, deps installed, dataset created, sentiment analyzed, recommender built."

# 6. Run the Streamlit YourNextAlbum interface app
echo "Starting Streamlit app..."
streamlit run album_recommender_prompt_app.py
