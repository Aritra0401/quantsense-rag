import os

folders = [
    'data/raw/filings',
    'data/raw/transcripts',
    'data/processed/chunks',
    'data/structured',
    'src/ingestion',
    'src/retrieval',
    'src/agent/tools',
    'src/guardrails',
    'src/features',
    'src/compression',
    'eval',
    'app/components',
    'scripts',
    '.chroma',
]

for f in folders:
    os.makedirs(f, exist_ok=True)

    if f.startswith('src'):
        open(f + '/__init__.py', 'a').close()

print("All folders created.")