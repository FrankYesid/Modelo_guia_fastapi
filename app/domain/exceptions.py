"""Errores del dominio.

Los mensajes (``message``) se muestran al cliente y por eso deben ser claros
pero sin informacion sensible. Los detalles tecnicos (``details``) son para el
desarrollador y quedan registrados en los logs del servidor.
"""

from __future__ import annotations


class DomainError(Exception):
    """Error base del dominio. Permite capturar todos los errores de negocio."""

    def __init__(self, message: str, *, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class ModelNotAvailableError(DomainError):
    """El modelo no esta cargado o no pudo cargarse al iniciar la aplicacion."""


class InvalidFeatureSetError(DomainError):
    """Las variables recibidas no coinciden con las que espera el modelo."""


class InferenceError(DomainError):
    """Fallo inesperado durante model.predict() / model.predict_proba()."""
