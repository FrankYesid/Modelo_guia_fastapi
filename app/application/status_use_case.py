"""Caso de uso: consultar el estado del modelo."""

from __future__ import annotations

from app.domain.entities import ModelInfo
from app.domain.ports import ModelGatewayPort


class GetStatusUseCase:
    """Devuelve los metadatos y el estado del modelo.

    Se usa en ``/health`` para distinguir entre "el servidor responde" y
    "el modelo realmente puede predecir".
    """

    def __init__(self, gateway: ModelGatewayPort) -> None:
        self._gateway = gateway

    def execute(self) -> ModelInfo:
        """Retorna la informacion actual del modelo (sin lanzar excepciones)."""
        return self._gateway.info
