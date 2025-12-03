#!/bin/zsh
set -e

# YourNextAlbum Quickstart script (runs from repo root even when called from scripts/)

# Resolve repository root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# 0. Create venv if neither .venv nor venv exists at repo root
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
  echo "Creating Python virtual environment (.venv) at $REPO_ROOT..."
  python3 -m venv .venv
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
python3 album_reviews_dataset_creator.py

# 4. Analyze sentiment of album reviews
echo "Running sentiment analysis..."
python3 album_reviews_text_sentiment_analyser.py

# 5. Build album recommender models
echo "Building recommender models..."
python3 album_recommender_model.py

echo "Quickstart complete: env ready, deps installed, dataset created, sentiment analyzed, recommender built."

# 6. Run the Streamlit YourNextAlbum interface app
echo "Starting Streamlit app..."
streamlit run album_recommender_prompt_app.py
