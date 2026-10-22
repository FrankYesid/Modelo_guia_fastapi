"""Proveedores de dependencias para las rutas de FastAPI.

Los casos de uso y la configuracion se crean **una sola vez** en
``app/main.py`` y se guardan en ``app.state``. Aqui solo se recuperan, de modo
que las rutas nunca instancian el modelo ni vuelven a cargarlo.
"""

from __future__ import annotations

from fastapi import Request

from app.application.predict_use_case import PredictUseCase
from app.application.status_use_case import GetStatusUseCase
from app.config import Settings


def get_settings(request: Request) -> Settings:
    """Devuelve la configuracion de la aplicacion."""
    return request.app.state.settings


def get_predict_use_case(request: Request) -> PredictUseCase:
    """Devuelve el caso de uso de prediccion."""
    return request.app.state.predict_use_case


def get_status_use_case(request: Request) -> GetStatusUseCase:
    """Devuelve el caso de uso de estado del modelo."""
    return request.app.state.status_use_case
