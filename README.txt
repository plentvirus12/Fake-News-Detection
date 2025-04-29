# Fake News Image Detection

## How to run:
1. Create virtual environment:
   python -m venv venv

2. Activate virtual environment:
   venv\Scripts\activate   (on Windows)

3. Install requirements:
   pip install -r requirements.txt

4. Run server:
   uvicorn app.main:app --reload

5. Open browser:
   http://127.0.0.1:8000

## Folder structure:
- app/
  ├─ main.py
  ├─ templates/index.html
  └─ static/ (optional)

- models/
  └─ hybrid_cnn_lstm_vit_model.h5
