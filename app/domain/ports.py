"""Puertos del hexagono.

Un **puerto** es una interfaz (contrato) que pertenece al dominio. Los
adaptadores del exterior se encargan de cumplirla:

- Adaptador de salida (driven): ``JoblibModelGateway`` carga el ``.pkl`` y
  ejecuta la inferencia con scikit-learn.
- Adaptador de entrada (driving): las rutas de FastAPI usan los casos de uso,
  que a su vez hablan con este puerto.

Usamos ``typing.Protocol`` para que el contrato sea estructural: cualquier
clase con los mismos metodos lo cumple, sin necesidad de heredar.
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from app.domain.entities import ModelInfo, Prediction


@runtime_checkable
class ModelGatewayPort(Protocol):
    """Contrato para obtener un modelo entrenado y usarlo para predecir."""

    def load(self) -> None:
        """Carga el modelo en memoria.

        Debe invocarse **una sola vez** al iniciar la aplicacion. No debe
        lanzar excepciones: si algo falla, guarda el motivo y deja el estado
        en ``info.loaded = False`` para que ``/health`` lo reporte.
        """
        ...

    @property
    def info(self) -> ModelInfo:
        """Metadatos y estado actual del modelo."""
        ...

    def predict(self, record: Mapping[str, float]) -> Prediction:
        """Ejecuta la inferencia sobre un unico registro.

        Args:
            record: Diccionario ``{nombre_variable: valor}`` con las variables
                que el modelo espera. El adaptador es responsable de respetar
                el orden y los nombres originales del entrenamiento.

        Raises:
            ModelNotAvailableError: Si el modelo no esta cargado.
            InferenceError: Si el modelo falla al predecir.
        """
        ...
