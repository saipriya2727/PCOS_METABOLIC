# Integrated ML-Based Screening System

PCOS screening for women and Metabolic Syndrome screening for men using XGBoost.

## Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
python train_models.py
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## Features
- XGBoost ML models (no rule-based logic)
- Multi-page React application
- PDF report downloads
- Responsive design
