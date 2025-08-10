# Tuberculosis Detection — Streamlit App

## What this project provides
- End-to-end pipeline: unzip → preprocess → split → optional augmentation → train → evaluate → predict.
- Choose among ResNet50, VGG16, EfficientNetB0 (transfer learning).
- Streamlit UI for training and inference.
- Files: `app.py`, `requirements.txt`, `deploy.sh`.

## Data format
Upload a ZIP containing `TB` and `NORMAL` folders anywhere inside (root or nested).

## Quickstart (local)
```bash
git clone <repo>
cd <repo>
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
