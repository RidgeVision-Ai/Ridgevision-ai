from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.ml.inference.predictor import MODEL_OUTPUT_LABELS, RidgeVisionPredictor


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_accepts_image():
    array = np.zeros((96, 96), dtype=np.uint8)
    array[:, ::4] = 255
    image = Image.fromarray(array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post("/predict", files={"file": ("fingerprint.png", buffer, "image/png")})

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"]
    assert payload["grad_cam_b64"].startswith("data:image/png;base64,")
    assert len(payload["all_probabilities"]) == 8


def test_trained_model_output_maps_last_neuron_to_o_positive():
    predictor = RidgeVisionPredictor()
    predictions = np.zeros(8, dtype=np.float32)
    predictions[7] = 1.0

    probabilities = predictor._probability_dict(predictions, MODEL_OUTPUT_LABELS)

    assert probabilities["O+"] == 100.0
    assert probabilities["O-"] == 0.0
