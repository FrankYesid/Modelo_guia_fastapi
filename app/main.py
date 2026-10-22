"""Punto de composicion (composition root) de la aplicacion.

Aqui se unen las piezas del hexagono y se define el punto de entrada que
Uvicorn necesita:

    uvicorn app.main:app --reload

Responsabilidades de este modulo:
    1. Construir la configuracion.
    2. Instanciar el adaptador de salida (modelo ``.pkl``).
    3. Instanciar los casos de uso con ese adaptador (inyeccion de dependencias).
    4. Cargar el modelo **una sola vez**, durante el arranque (lifespan).
    5. Registrar la traduccion de errores de dominio a codigos HTTP.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.adapters.inbound.http.routes import router
from app.adapters.outbound.joblib_model_gateway import JoblibModelGateway
from app.application.predict_use_case import PredictUseCase
from app.application.status_use_case import GetStatusUseCase
from app.config import Settings
from app.domain.exceptions import (
    DomainError,
    InferenceError,
    InvalidFeatureSetError,
    ModelNotAvailableError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

#: Traduccion de errores de dominio a codigos HTTP.
HTTP_STATUS_BY_ERROR: tuple[tuple[type[DomainError], int], ...] = (
    (InvalidFeatureSetError, 422),
    (ModelNotAvailableError, 503),
    (InferenceError, 500),
)


def _register_exception_handlers(app: FastAPI) -> None:
    """Convierte los errores de dominio en respuestas JSON controladas.

    Al cliente solo se le envia ``detail`` (mensaje legible) y ``error``
    (nombre del tipo). Las trazas y rutas internas se quedan en los logs.
    """

    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        status_code = 500
        for error_type, code in HTTP_STATUS_BY_ERROR:
            if isinstance(exc, error_type):
                status_code = code
                break

        logger.warning(
            "Error de dominio %s en %s %s | detalle interno: %s",
            type(exc).__name__,
            request.method,
            request.url.path,
            exc.details,
        )
        return JSONResponse(
            status_code=status_code,
            content={"detail": exc.message, "error": type(exc).__name__},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Fabrica de la aplicacion.

    Recibe la configuracion como parametro (por defecto la lee del entorno),
    lo que permite inyectar un modelo distinto en las pruebas automatizadas.

    Args:
        settings: Configuracion a usar. Si es ``None`` se usa
            :meth:`Settings.from_env`.

    Returns:
        La instancia de ``FastAPI`` lista para ser servida por Uvicorn.
    """
    settings = settings or Settings.from_env()

    # Adaptador de salida (driven adapter).
    gateway = JoblibModelGateway(
        model_path=settings.model_path,
        model_name=settings.model_name,
        model_version=settings.model_version,
    )

    predict_use_case = PredictUseCase(gateway)
    status_use_case = GetStatusUseCase(gateway)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Carga el modelo al arrancar y lo libera al detener el servidor."""
        logger.info("Cargando el modelo una unica vez (ruta configurada: %s)", settings.model_path)
        gateway.load()
        if not gateway.info.loaded:
            logger.warning(
                "La API arrancara en modo degradado: /health lo reportara y "
                "/predict devolvera 503 hasta que el modelo este disponible."
            )
        yield
        logger.info("Servidor detenido.")

    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        lifespan=lifespan,
    )

    # Contenedor de dependencias: se resuelven una sola vez.
    app.state.settings = settings
    app.state.gateway = gateway
    app.state.predict_use_case = predict_use_case
    app.state.status_use_case = status_use_case

    _register_exception_handlers(app)
    app.include_router(router)

    return app


#: Instancia que Uvicorn importa: ``uvicorn app.main:app``
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    _settings = Settings.from_env()
    uvicorn.run("app.main:app", host=_settings.host, port=_settings.port, reload=True)
