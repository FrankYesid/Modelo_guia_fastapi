"""Entidades y objetos de valor del dominio.

Son estructuras inmutables y "puras": representan el resultado de una
prediccion y el estado del modelo, sin depender de ningun framework.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Estado y metadatos del modelo cargado.

    Attributes:
        name: Nombre logico del modelo (configurable desde el entorno).
        version: Version del modelo (configurable desde el entorno).
        loaded: ``True`` solo si el modelo esta en memoria y listo para predecir.
        feature_names: Nombres y **orden** de las variables que espera el modelo.
        classes: Clases que puede predecir (vacio si el modelo no las expone).
        supports_probabilities: ``True`` si el modelo implementa ``predict_proba``.
        error: Motivo legible si ``loaded`` es ``False`` (sin rutas ni trazas).
    """

    name: str
    version: str
    loaded: bool
    feature_names: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()
    supports_probabilities: bool = False
    error: str | None = None


@dataclass(frozen=True)
class Prediction:
    """Resultado de una inferencia sobre un unico registro de entrada.

    Attributes:
        label: Clase predicha, siempre convertida a texto para poder
            serializarse como JSON.
        probabilities: Probabilidad por clase (suma 1.0) o ``None`` si el
            modelo no implementa ``predict_proba``.
    """

    label: str
    probabilities: dict[str, float] | None = None
