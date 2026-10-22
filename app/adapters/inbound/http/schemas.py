"""Esquemas Pydantic del adaptador HTTP (entrada y salida).

ADAPTAR AL MODELO REAL: los nombres, tipos, rangos y cantidad de variables
dependen **exclusivamente** del modelo entrenado. Los valores de este archivo
corresponden al modelo de ejemplo del proyecto: un ``Pipeline`` de
scikit-learn (``StandardScaler`` + ``LogisticRegression``) entrenado sobre el
conjunto de datos **Iris** (ver ``notebooks/02_entrenamiento_modelo.ipynb``).

Si entrenas con otro conjunto de datos:
    1. Ejecuta ``model.feature_names_in_`` para conocer los nombres y su orden.
    2. Renombra los campos de :class:`IrisFeatures` con esos nombres exactos.
    3. Ajusta los rangos (``ge``/``le``) al dominio observado en el entrenamiento.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: Ejemplo valido usado en la documentacion automática (Swagger UI y ReDoc).
EXAMPLE_REQUEST = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


class IrisFeatures(BaseModel):
    """Cuerpo de la solicitud ``POST /predict`` (un unico registro).

    Todas las variables son numericas porque el modelo de ejemplo fue entrenado
    solo con variables numericas.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": EXAMPLE_REQUEST},
    )

    sepal_length: float = Field(
        ...,
        ge=4.0,
        le=8.0,
        description="Longitud del sepalo en cm (rango observado en el entrenamiento).",
    )
    sepal_width: float = Field(
        ...,
        ge=2.0,
        le=4.5,
        description="Ancho del sepalo en cm (rango observado en el entrenamiento).",
    )
    petal_length: float = Field(
        ...,
        ge=1.0,
        le=7.0,
        description="Longitud del petalo en cm (rango observado en el entrenamiento).",
    )
    petal_width: float = Field(
        ...,
        ge=0.1,
        le=2.5,
        description="Ancho del petalo en cm (rango observado en el entrenamiento).",
    )


class PredictResponse(BaseModel):
    """Respuesta de ``POST /predict``."""

    prediction: str = Field(..., description="Clase predicha por el modelo.")
    probabilities: dict[str, float] | None = Field(
        default=None,
        description=(
            "Probabilidad por clase. Es null cuando el modelo no implementa "
            "predict_proba()."
        ),
    )
    model_name: str = Field(..., description="Nombre logico del modelo.")
    model_version: str = Field(..., description="Version del modelo.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "setosa",
                "probabilities": {
                    "setosa": 0.981234,
                    "versicolor": 0.015432,
                    "virginica": 0.003334,
                },
                "model_name": "iris-classifier",
                "model_version": "1.0.0",
            }
        }
    )


class HealthResponse(BaseModel):
    """Respuesta de ``GET /health``.

    ``status`` describe el **servidor**; ``model_loaded`` describe el **modelo**.
    Un servidor activo con ``model_loaded = false`` significa que la API
    responde pero no puede predecir.
    """

    status: str = Field(..., description="ok si el modelo esta cargado; degraded si no lo esta.")
    model_loaded: bool = Field(..., description="True si el modelo esta en memoria y puede predecir.")
    model_name: str = Field(..., description="Nombre logico del modelo.")
    model_version: str = Field(..., description="Version del modelo.")
    supports_probabilities: bool = Field(..., description="True si el modelo implementa predict_proba().")
    features: list[str] = Field(default_factory=list, description="Variables esperadas por el modelo.")
    classes: list[str] = Field(default_factory=list, description="Clases que puede predecir.")
    detail: str | None = Field(
        default=None,
        description="Motivo legible si el modelo no esta cargado (sin rutas ni trazas internas).",
    )


class RootResponse(BaseModel):
    """Respuesta de ``GET /``."""

    project: str
    description: str
    version: str
    status: str = Field(..., description="Estado general de la API.")
    documentation: dict[str, str] = Field(
        ..., description="Rutas de la documentacion interactiva."
    )


class ErrorResponse(BaseModel):
    """Formato de error comun a toda la API."""

    detail: str = Field(..., description="Mensaje legible para el cliente.")
    error: str = Field(..., description="Tipo de error del dominio.")
