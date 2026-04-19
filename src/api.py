"""
FastAPI REST API for skin cancer prediction.

Provides a /predict endpoint that accepts dermoscopic images
and returns classification results with confidence scores.
Includes auto-generated Swagger/OpenAPI documentation.
"""

import os
import io
import time
import logging
import glob
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from src.inference import SkinCancerPredictor
from src.monitoring import (
    setup_prometheus,
    track_prediction,
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    MODEL_CONFIDENCE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Available model names (must match checkpoint filename pattern: {name}_best.pth)
SUPPORTED_MODELS = ["resnet50", "efficientnet", "vit"]

# Global predictors registry: model_name -> SkinCancerPredictor
predictors: Dict[str, SkinCancerPredictor] = {}
default_model: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all supported models on startup (with checkpoint if available)."""
    global predictors, default_model

    for model_name in SUPPORTED_MODELS:
        ckpt_path = os.path.join("models", f"{model_name}_best.pth")
        has_checkpoint = os.path.isfile(ckpt_path)
        logger.info(
            f"Loading model: {model_name}"
            + (f" from {ckpt_path}" if has_checkpoint else " (default weights)")
        )
        predictors[model_name] = SkinCancerPredictor(
            model_name=model_name,
            checkpoint_path=ckpt_path if has_checkpoint else None,
        )
        logger.info(f"Model {model_name} loaded successfully")

    # Default: prefer env var, then efficientnet
    env_model = os.getenv("MODEL_NAME")
    if env_model and env_model in predictors:
        default_model = env_model
    else:
        default_model = "efficientnet"

    logger.info(f"Available models: {list(predictors.keys())}")
    logger.info(f"Default model: {default_model}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Skin Cancer Classification API",
    description=(
        "REST API for classifying dermoscopic images of skin lesions.\n\n"
        "Uses deep learning models (ResNet50, EfficientNet, ViT) "
        "trained on the ISIC Skin Cancer dataset to assist dermatologists "
        "in detecting potential malignancies.\n\n"
        "**Important**: This is a decision support tool, not a diagnostic device. "
        "Results must be reviewed by qualified medical professionals."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Setup Prometheus metrics
setup_prometheus(app)


# --- Response Models ---


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    model_name: str
    is_malignant: bool
    risk_level: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_class": "nevus",
                    "confidence": 0.8523,
                    "class_probabilities": {
                        "actinic keratosis": 0.0121,
                        "basal cell carcinoma": 0.0234,
                        "dermatofibroma": 0.0098,
                        "melanoma": 0.0521,
                        "nevus": 0.8523,
                        "pigmented benign keratosis": 0.0112,
                        "seborrheic keratosis": 0.0089,
                        "squamous cell carcinoma": 0.0179,
                        "vascular lesion": 0.0123,
                    },
                    "model_name": "efficientnet",
                    "is_malignant": False,
                    "risk_level": "LOW",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    default_model: Optional[str]


# --- Endpoints ---


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Skin Cancer Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint for monitoring and orchestration."""
    return HealthResponse(
        status="healthy" if predictors else "unhealthy",
        models_loaded=list(predictors.keys()),
        default_model=default_model,
    )


@app.get("/models", tags=["General"])
async def list_models():
    """List available models and which is the default."""
    return {
        "available_models": list(predictors.keys()),
        "default_model": default_model,
        "supported_models": SUPPORTED_MODELS,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Classify a dermoscopic image",
    description=(
        "Upload a dermoscopic image to classify the skin lesion type.\n\n"
        "Use the **model_name** query parameter to choose the model "
        "(resnet50, efficientnet, vit). "
        "Omit to use the server default.\n\n"
        "Supported classes (9):\n"
        "- **actinic keratosis**: Pre-malignant skin lesion\n"
        "- **basal cell carcinoma**: Malignant skin cancer\n"
        "- **dermatofibroma**: Benign skin growth\n"
        "- **melanoma**: Malignant skin cancer\n"
        "- **nevus**: Common mole (benign)\n"
        "- **pigmented benign keratosis**: Benign keratosis\n"
        "- **seborrheic keratosis**: Benign skin growth\n"
        "- **squamous cell carcinoma**: Malignant skin cancer\n"
        "- **vascular lesion**: Benign vascular growth"
    ),
)
async def predict(
    file: UploadFile = File(..., description="Dermoscopic image file (JPEG/PNG)"),
    model_name: Optional[str] = Query(
        default=None,
        description="Model to use for prediction. Options: resnet50, efficientnet, vit. Defaults to the server's default model.",
    ),
):
    """Classify a dermoscopic skin lesion image."""
    if not predictors:
        raise HTTPException(status_code=503, detail="No models loaded")

    # Resolve which model to use
    selected_model = model_name or default_model
    if selected_model not in predictors:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{selected_model}' not available. Available: {list(predictors.keys())}",
        )
    predictor = predictors[selected_model]

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}",
        )

    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Predict with timing
    start_time = time.time()
    result = predictor.predict(image)
    latency = time.time() - start_time

    # Track metrics
    track_prediction(result["predicted_class"], result["confidence"], latency)

    logger.info(
        f"Prediction: {result['predicted_class']} "
        f"(confidence={result['confidence']:.4f}, "
        f"risk={result['risk_level']}, "
        f"latency={latency:.3f}s)"
    )

    return PredictionResponse(**result)


@app.get("/classes", tags=["General"])
async def get_classes():
    """Get the list of supported skin lesion classes."""
    descriptions = {
        "actinic keratosis": "Pre-malignant keratotic lesion (actinic keratosis)",
        "basal cell carcinoma": "Malignant skin cancer, rarely metastasizes",
        "dermatofibroma": "Benign fibrous skin growth",
        "melanoma": "Malignant skin cancer - most dangerous type",
        "nevus": "Common mole (melanocytic nevus) - benign",
        "pigmented benign keratosis": "Benign pigmented keratotic lesion",
        "seborrheic keratosis": "Benign waxy skin growth",
        "squamous cell carcinoma": "Malignant skin cancer, can metastasize",
        "vascular lesion": "Benign vascular growth (angioma, angiokeratoma)",
    }
    return {"classes": descriptions}
