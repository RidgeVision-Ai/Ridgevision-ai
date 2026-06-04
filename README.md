# RidgeVision AI

Experimental AI-powered dermatoglyphic analysis platform for fingerprint-based ABO/Rh pattern estimation using deep learning, texture feature extraction, and explainable AI.

> Important: this project is a research and portfolio prototype only. Dermatoglyphic blood group estimation is not clinically validated, and this software must not be used for diagnosis or medical decision-making.

## Features

- FastAPI inference API with `/health` and `/predict`
- Fingerprint preprocessing with CLAHE, denoising, Gabor ridge enhancement, and normalization
- Handcrafted texture feature extraction using LBP, GLCM-style statistics, and ridge density
- Research-mode inference engine with a clean adapter point for TensorFlow/Keras models
- Explainability heatmap generation returned as base64 PNG
- Responsive biomedical dashboard with drag-and-drop upload, probability matrix, and heatmap preview

## Project Structure

```text
.
|-- backend/
|   |-- api/
|   |-- core/
|   |-- ml/
|   |   |-- explainability/
|   |   |-- feature_engineering/
|   |   |-- inference/
|   |   |-- models/
|   |   |-- preprocessing/
|   |   `-- training/
|   |-- main.py
|   `-- requirements.txt
|-- datasets/
|-- docker/
|-- frontend/
|   |-- index.html
|   |-- js/
|   `-- styles/
|-- tests/
`-- README.md
```

## Datasets

Use the labeled Kaggle dataset for supervised ABO/Rh experiments:

- <https://www.kaggle.com/datasets/rajumavinmar/finger-print-based-blood-group-dataset>

Use the original SOCOFing dataset for unlabeled fingerprint research, quality checks, augmentation, or self-supervised pretraining:

- <https://www.kaggle.com/datasets/ruizgara/socofing>

Suggested local layout is documented in `datasets/README.md`.

## Run Locally

Python 3.11 or 3.12 is recommended for the scientific imaging stack. You can run the project directly without creating a virtual environment.

```bash
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Then open `frontend/index.html` in a browser. The frontend posts to `http://127.0.0.1:8000/predict` by default.

## Run In Colab Or Kaggle

Use [notebooks/ridgevision_colab_kaggle_demo.ipynb](notebooks/ridgevision_colab_kaggle_demo.ipynb) for hosted notebook execution.

### Colab

1. Upload or clone this project into Colab.
2. Open `notebooks/ridgevision_colab_kaggle_demo.ipynb`.
3. Run the dependency install cell.
4. Upload a fingerprint image when prompted.
5. Call `predict_fingerprint(image_path)` to view the predicted class, probabilities, extracted features, and heatmap.

### Kaggle

1. Create a Kaggle Notebook.
2. Add this project as notebook files or clone it into the working directory.
3. Enable internet if dependencies need to be installed.
4. Add your fingerprint image dataset under `/kaggle/input/...`.
5. Run:

```python
result = predict_fingerprint("/kaggle/input/your-dataset/fingerprint.png")
```

The notebook path does not require running the frontend. It calls the same ML pipeline directly from Python, which is usually the simplest setup for Colab/Kaggle experiments.

## API

### `POST /predict`

Upload a fingerprint image as multipart form data:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@fingerprint.png"
```

Response:

```json
{
  "predicted_class": "A+",
  "confidence": 18.42,
  "all_probabilities": {
    "A+": 18.42
  },
  "grad_cam_b64": "data:image/png;base64,...",
  "features": {},
  "disclaimer": "Experimental research output only. Not for clinical use."
}
```

## Model Integration

The current inference engine intentionally uses deterministic research-mode scoring so the app can run without private training data or model weights. To connect a trained model, replace or extend `backend/ml/inference/predictor.py` and use `backend/ml/models/architecture.py` as the architecture starting point.

For serious experiments, train and validate on a properly consented dataset, save the trained Keras model under `models/`, and update the predictor to load those weights instead of using research-mode scoring.

## License

MIT License
