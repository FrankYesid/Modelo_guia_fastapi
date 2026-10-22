"""Rutas HTTP de la API.

Este es el adaptador de entrada: traduce HTTP <-> casos de uso. No contiene
logica de negocio ni detalles de scikit-learn.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.adapters.inbound.http.dependencies import (
    get_predict_use_case,
    get_settings,
    get_status_use_case,
)
from app.adapters.inbound.http.schemas import (
    ErrorResponse,
    HealthResponse,
    IrisFeatures,
    PredictResponse,
    RootResponse,
)
from app.application.predict_use_case import PredictUseCase
from app.application.status_use_case import GetStatusUseCase
from app.config import Settings

router = APIRouter()


@router.get(
    "/",
    response_model=RootResponse,
    status_code=status.HTTP_200_OK,
    summary="Bienvenida",
    description="Presenta el proyecto, su proposito y su estado general.",
    tags=["General"],
)
def read_root(settings: Settings = Depends(get_settings)) -> RootResponse:
    """Devuelve informacion basica de la API."""
    return RootResponse(
        project=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        status="ok",
        documentation={
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Estado de la aplicacion y del modelo",
    description=(
        "Informa si el servidor esta activo y, de forma independiente, si el "
        "modelo esta cargado y puede predecir. Devuelve 200 siempre que el "
        "servidor responda; revisa 'model_loaded' para saber si /predict "
        "funcionara."
    ),
    tags=["General"],
)
def health(status_use_case: GetStatusUseCase = Depends(get_status_use_case)) -> HealthResponse:
    """Reporta el estado del modelo cargado en memoria."""
    info = status_use_case.execute()
    return HealthResponse(
        status="ok" if info.loaded else "degraded",
        model_loaded=info.loaded,
        model_name=info.name,
        model_version=info.version,
        supports_probabilities=info.supports_probabilities,
        features=list(info.feature_names),
        classes=list(info.classes),
        detail=info.error,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Generar una prediccion",
    description=(
        "Recibe un registro en JSON, valida los campos y tipos, comprueba que "
        "las variables coincidan con las del modelo y devuelve la clase "
        "predicha junto con las probabilidades (si el modelo las soporta)."
    ),
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Datos invalidos o variables que no coinciden con el modelo.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": "El modelo no esta cargado: la API responde pero no puede predecir.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Fallo controlado durante la inferencia.",
        },
    },
    tags=["Modelo"],
)
def predict(
    payload: IrisFeatures,
    settings: Settings = Depends(get_settings),
    predict_use_case: PredictUseCase = Depends(get_predict_use_case),
) -> PredictResponse:
    """Ejecuta el caso de uso de prediccion y construye la respuesta."""
    prediction = predict_use_case.execute(payload.model_dump())

    return PredictResponse(
        prediction=prediction.label,
        probabilities=prediction.probabilities,
        model_name=settings.model_name,
        model_version=settings.model_version,
    )
