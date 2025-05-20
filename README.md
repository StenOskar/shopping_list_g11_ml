# Shopping List ML Model

This repository contains the machine-learning service that powers the 
ShelfAware mobile application. It trains personalised grocery-recommendation 
models from each user’s purchase history, then publishes the predictions back to 
Supabase so the Flutter app can surface them in Shopping Suggestions.

## Features
- Collaborative Filtering model that learns similarities 
  between users and products.
- K‑Nearest‑Neighbour clustering that groups similar items to smooth the recommendations.
- Automatic data ingestion from the purchase_history and inventory tables in Supabase.
- Periodic batch inference – writes the top‑N item IDs into the recommendations
  table in Supabase.
- One‑command training (python train.py) and one‑command deployment (python save_recommendation.py)

## Prerequisites
- A Supabase project with the following tables: receipts, receipt_items, inventory, profiles, recommendations
- Service role key for your project (allows insert/update during batch inference)

## Installation
### 1 Clone & enter the repo
- $ git clone https://github.com/shopping‑list‑ml.git
- $ cd shopping‑list‑ml

### 2 Create & activate a virtual environment
- $ python -m venv .venv
- $ source .venv/bin/activate

### 3 Install dependencies
$ pip install -r requirements.txt

## Configuration
All secrets are read from secrets.env.  Create the file at the repo root:
- SUPABASE_URL=https://your‑project.supabase.co
- SUPABASE_SERVICE_ROLE=ey...YourServiceRoleKey

## Usage

### Training
- $ python train.py
- Artifacts are saved to weights/

### Batch deployment
- Compute top‑10 items per user and upsert into Supabase
- $ python save_recommendation.py
