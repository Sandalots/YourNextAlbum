# 🎵 Album Recommendation AI System
AI-powered album recommender using Pitchfork reviews with advanced NLP, sentiment analysis, and semantic search.

## Quick Start
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python album_scraper.py          # Scrape reviews
python dataset_creator.py        # Preprocess with NLTK and turn into a dataset
python sentiment_analyzer.py     # Analyze (8+ feature types) from review texts
python recommender_model.py      # builds the model and recommends albums based on given prompt
streamlit run app.py             # Launches a streamlit UI to allow users to prompt the created model for album recommendations
```

## Analysis Features
- **Sentiment & Consensus** - Tone + score alignment
- **Instrumentation** - Guitar, synth, drums, vocals, etc.
- **Mood & Energy** - Sad/happy/dark + high/medium/low
- **Production** - Polished/raw/experimental, analog/digital
- **Novelty** - Innovative vs. derivative detection
- **Context** - Party, study, workout, relaxation, etc.
- **Temporal** - Vintage/contemporary/timeless
- **Polarizing** - Divisive "love it or hate it" albums
