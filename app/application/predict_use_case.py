"""Caso de uso: generar una prediccion."""

from __future__ import annotations

from typing import Mapping, Sequence

from app.domain.entities import Prediction
from app.domain.exceptions import (
    DomainError,
    InferenceError,
    InvalidFeatureSetError,
    ModelNotAvailableError,
)
from app.domain.ports import ModelGatewayPort


class PredictUseCase:
    """Valida la entrada y delega la inferencia en el puerto del modelo.

    Pasos que ejecuta:
        1. Verifica que el modelo este cargado.
        2. Comprueba que las variables recibidas coincidan con las esperadas.
        3. Delega ``predict`` en el adaptador y traduce fallos inesperados a
           :class:`InferenceError`.
    """

    def __init__(self, gateway: ModelGatewayPort) -> None:
        self._gateway = gateway

    def execute(self, record: Mapping[str, float]) -> Prediction:
        """Ejecuta el caso de uso para un unico registro.

        Args:
            record: Variables de entrada ya validadas por Pydantic.

        Returns:
            La prediccion y, si el modelo lo permite, sus probabilidades.

        Raises:
            ModelNotAvailableError: El modelo no esta cargado (HTTP 503).
            InvalidFeatureSetError: Las variables no coinciden (HTTP 422).
            InferenceError: El modelo fallo al predecir (HTTP 500).
        """
        info = self._gateway.info
        if not info.loaded:
            raise ModelNotAvailableError(
                "El modelo no esta disponible en este momento. "
                "Consulte GET /health para mas informacion.",
                details=info.error,
            )

        self._validate_features(record, info.feature_names)

        try:
            return self._gateway.predict(record)
        except DomainError:
            raise
        except Exception as exc:  # pragma: no cover - salvaguarda
            raise InferenceError(
                "No fue posible generar la prediccion con el modelo cargado.",
                details=f"{type(exc).__name__}: {exc}",
            ) from exc

    @staticmethod
    def _validate_features(record: Mapping[str, float], expected: Sequence[str]) -> None:
        """Comprueba que el registro contenga exactamente las variables del modelo.

        Nota: el **orden** no se valida aqui porque el adaptador de salida
        reconstruye las columnas en el orden original del entrenamiento.
        """
        received = set(record)
        expected_set = set(expected)

        missing = [name for name in expected if name not in received]
        unexpected = sorted(received - expected_set)

        if missing or unexpected:
            partes: list[str] = []
            if missing:
                partes.append(f"faltan: {', '.join(missing)}")
            if unexpected:
                partes.append(f"no reconocidas: {', '.join(unexpected)}")
            raise InvalidFeatureSetError(
                "Las variables enviadas no coinciden con las que espera el modelo "
                f"({'; '.join(partes)}).",
                details=f"Variables esperadas: {list(expected)}",
            )
