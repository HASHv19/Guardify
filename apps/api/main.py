
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from guardify.inference import InferenceService

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("guardify.api")

# Global reference for the loaded model service
_service_instance: InferenceService | None = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Text to classify. Max 2000 chars.")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    flagged_tokens: list[str]
    normalized_text: str
    model_version: str
    sub_category: str | None = None
    needs_review: bool = False


def _bundle_path() -> Path | None:
    candidate = os.getenv("GUARDIFY_MODEL_BUNDLE")
    if not candidate:
        return None
    path = Path(candidate).expanduser().resolve()
    return path if path.exists() else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service_instance
    bundle = _bundle_path()
    if not bundle:
        logger.error("Startup failed: GUARDIFY_MODEL_BUNDLE is not set or the path does not exist.")
        raise RuntimeError("GUARDIFY_MODEL_BUNDLE is not configured properly.")
    
    logger.info(f"Loading Guardify model bundle from: {bundle}")
    try:
        _service_instance = InferenceService(bundle)
        logger.info(f"Successfully loaded model: {_service_instance.model_name} (Type: {_service_instance.model_type})")
    except Exception as e:
        logger.error(f"Failed to load model bundle: {e}")
        raise RuntimeError(f"Failed to initialize InferenceService: {e}") from e
    
    yield
    logger.info("Shutting down Guardify API...")
    _service_instance = None


def get_service() -> InferenceService:
    if _service_instance is None:
        raise HTTPException(status_code=503, detail="Model service is unavailable or failed to load.")
    return _service_instance


def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    
    Request Flow for /predict:
    1. HTTP POST Request arrives.
    2. FastAPI validates payload against PredictRequest schema (e.g., length).
    3. Payload text is passed to InferenceService.predict_text().
    4. Text is processed via PreprocessingPipeline (normalization, token flagging).
    5. Extracted features are sent to the underlying model (SVM or Transformer) for inference.
    6. Raw outputs are converted to probabilities and a final class label.
    7. PredictResponse schema structures the result and returns it.
    """
    app = FastAPI(title="Guardify API", version="1.0.0", lifespan=lifespan)
    
    cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        # Safe logging: intentionally avoids logging the raw text payload to prevent sensitive data leaks
        logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - {process_time:.2f}ms")
        return response

    @app.get("/health")
    def health() -> dict[str, object]:
        # Return generic health status. We avoid exposing absolute paths in production, 
        # but for this academic demo, showing the bundle_path is requested.
        try:
            service = get_service()
            return {
                "status": "ok",
                "bundle_loaded": True,
                "model_name": service.model_name,
                "model_type": service.model_type,
                "device": service.device,
                "bundle_path": str(_bundle_path()),
            }
        except HTTPException:
            return {
                "status": "degraded",
                "bundle_loaded": False,
                "bundle_path": str(_bundle_path()) if _bundle_path() else None,
            }

    @app.get("/model-info")
    def model_info() -> dict[str, object]:
        service = get_service()
        return service.model_info()

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        service = get_service()
        try:
            return PredictResponse(**service.predict_text(payload.text))
        except ValueError as exc:
            # Mask internal stack traces and return a clean 422 error
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
