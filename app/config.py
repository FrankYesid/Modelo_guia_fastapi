"""Configuracion central de la aplicacion.

Todos los valores se pueden sobrescribir con variables de entorno, lo que
permite usar la misma imagen/codigo en local, en pruebas y en el hosting sin
tocar el codigo fuente.

Este modulo es deliberadamente "tonto": no importa FastAPI ni scikit-learn.
Es un objeto plano que se inyecta en los adaptadores desde el exterior
(composicion de dependencias en ``app/main.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Raiz del proyecto (la carpeta que contiene ``app/``, ``models/``, ``tests/``).
BASE_DIR = Path(__file__).resolve().parent.parent

#: Ruta por defecto del modelo entrenado.
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "modelo.pkl"


def _env_str(name: str, default: str) -> str:
    """Lee una variable de entorno de texto; si esta vacia usa el valor por defecto."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    """Lee una variable de entorno entera y falla rapido si el valor no es valido."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:  # pragma: no cover - depende del entorno del usuario
        raise ValueError(
            f"La variable de entorno {name} debe ser un numero entero, se recibio: {raw!r}"
        ) from exc


@dataclass(frozen=True)
class Settings:
    """Configuracion inmutable de la API.

    Se construye normalmente con :meth:`Settings.from_env`.
    """

    # Metadatos que se muestran en / y en la documentacion automatica.
    app_name: str = "API del modelo de Machine Learning"
    app_description: str = (
        "API REST construida con FastAPI que carga un modelo de Machine Learning "
        "previamente entrenado (archivo .pkl) y expone predicciones por HTTP. "
        "Proyecto base con arquitectura hexagonal para fines didacticos."
    )
    app_version: str = "0.1.0"

    # Ubicacion y metadatos del modelo serializado.
    model_path: Path = DEFAULT_MODEL_PATH
    model_name: str = "iris-classifier"
    model_version: str = "1.0.0"

    # Parametros del servidor ASGI (Uvicorn).
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        """Construye la configuracion leyendo variables de entorno.

        Variables reconocidas: ``APP_NAME``, ``APP_VERSION``, ``MODEL_PATH``,
        ``MODEL_NAME``, ``MODEL_VERSION``, ``HOST``, ``PORT``.
        """
        return cls(
            app_name=_env_str("APP_NAME", cls.app_name),
            app_version=_env_str("APP_VERSION", cls.app_version),
            model_path=Path(_env_str("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser(),
            model_name=_env_str("MODEL_NAME", cls.model_name),
            model_version=_env_str("MODEL_VERSION", cls.model_version),
            host=_env_str("HOST", cls.host),
            port=_env_int("PORT", cls.port),
        )
