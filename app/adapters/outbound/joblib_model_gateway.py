"""Adaptador de salida: carga y ejecuta el modelo serializado en un ``.pkl``.

Implementa el puerto :class:`~app.domain.ports.ModelGatewayPort` usando
``joblib`` + ``scikit-learn``. Es la unica capa que conoce esas librerias.

Dos escenarios contemplados:

**Escenario 1 - el ``.pkl`` contiene solo el estimador entrenado.**
    Aqui se asume que el archivo ya incluye *todas* las transformaciones que
    deben reproducirse (escalado, codificacion, seleccion de variables). Si no
    fuera asi, el escalador/codificador deberia guardarse junto al estimador y
    aplicarse en este mismo punto, **nunca** reajustarse con datos nuevos.

**Escenario 2 (recomendado) - el ``.pkl`` contiene un ``Pipeline`` completo.**
    ``model.predict()`` aplica el ``StandardScaler`` y el estimador con los
    parametros aprendidos durante el entrenamiento. Esta es la situacion de
    ``models/modelo.pkl``, generado por ``notebooks/02_entrenamiento_modelo.ipynb``.

AVISO DE SEGURIDAD: los archivos Pickle pueden ejecutar codigo arbitrario al
deserializarse. Carga unicamente archivos generados por ti o por una fuente
confiable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from app.domain.entities import ModelInfo, Prediction
from app.domain.exceptions import InferenceError, ModelNotAvailableError

logger = logging.getLogger(__name__)


class JoblibModelGateway:
    """Carga un modelo ``.pkl`` una sola vez y ejecuta la inferencia.

    Args:
        model_path: Ruta al archivo ``.pkl``.
        model_name: Nombre logico del modelo (se reporta en la API).
        model_version: Version del modelo (se reporta en la API).
    """

    def __init__(self, model_path: Path | str, model_name: str, model_version: str) -> None:
        self._model_path = Path(model_path)
        self._model_name = model_name
        self._model_version = model_version

        self._model: Any | None = None
        self._feature_names: tuple[str, ...] = ()
        self._classes: tuple[str, ...] = ()
        self._supports_probabilities: bool = False
        self._load_error: str | None = None

    # ------------------------------------------------------------------ #
    # Puerto: carga del modelo
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Carga el modelo en memoria.

        No propaga excepciones: si el archivo no existe o no puede
        deserializarse, guarda un motivo legible y deja el adaptador en estado
        "no disponible" para que ``/health`` lo informe y ``/predict`` devuelva
        HTTP 503. Asi el servidor arranca igual y el error es explicito.
        """
        self._reset()

        if not self._model_path.is_file():
            self._load_error = "Archivo del modelo no encontrado."
            logger.error(
                "No se encontro el archivo del modelo. Copie el .pkl entrenado en: %s",
                self._model_path,
            )
            return

        try:
            model = joblib.load(self._model_path)
        except Exception as exc:
            self._load_error = "No fue posible deserializar el archivo del modelo."
            logger.error(
                "Error al cargar el modelo (%s). Verifique que scikit-learn/joblib tengan "
                "las mismas versiones usadas durante el entrenamiento. Detalle: %s: %s",
                self._model_path,
                type(exc).__name__,
                exc,
            )
            return

        self._model = model
        self._feature_names = self._resolve_feature_names(model)
        self._classes = self._resolve_classes(model)
        self._supports_probabilities = callable(getattr(model, "predict_proba", None))
        self._load_error = None

        logger.info(
            "Modelo '%s' v%s cargado desde %s | variables=%d | clases=%d | predict_proba=%s",
            self._model_name,
            self._model_version,
            self._model_path,
            len(self._feature_names),
            len(self._classes),
            self._supports_probabilities,
        )

    # ------------------------------------------------------------------ #
    # Puerto: metadatos
    # ------------------------------------------------------------------ #
    @property
    def info(self) -> ModelInfo:
        """Estado y metadatos del modelo (seguros para exponer por HTTP)."""
        return ModelInfo(
            name=self._model_name,
            version=self._model_version,
            loaded=self._model is not None,
            feature_names=self._feature_names,
            classes=self._classes,
            supports_probabilities=self._supports_probabilities,
            error=self._load_error,
        )

    # ------------------------------------------------------------------ #
    # Puerto: inferencia
    # ------------------------------------------------------------------ #
    def predict(self, record: Mapping[str, float]) -> Prediction:
        """Ejecuta ``predict`` (y ``predict_proba`` si existe) sobre un registro.

        El ``DataFrame`` se construye con ``columns=self._feature_names``, es
        decir, con los nombres y el orden originales del entrenamiento, sin
        renombrar ni reordenar variables.
        """
        if self._model is None:
            raise ModelNotAvailableError(
                "El modelo no esta disponible en este momento.",
                details=self._load_error,
            )

        features = list(self._feature_names)
        frame = pd.DataFrame(
            [[record[name] for name in features]],
            columns=features,
        )

        try:
            raw_labels = np.atleast_1d(self._model.predict(frame))
        except Exception as exc:
            logger.exception("Fallo model.predict()")
            raise InferenceError(
                "El modelo no pudo procesar los datos enviados.",
                details=f"{type(exc).__name__}: {exc}",
            ) from exc

        label = self._to_json_scalar(raw_labels[0])
        probabilities = self._predict_probabilities(frame)

        return Prediction(label=label, probabilities=probabilities)

    # ------------------------------------------------------------------ #
    # Utilidades internas
    # ------------------------------------------------------------------ #
    def _predict_probabilities(self, frame: pd.DataFrame) -> dict[str, float] | None:
        """Devuelve ``{clase: probabilidad}`` o ``None`` si no aplica.

        No todos los modelos implementan ``predict_proba`` (por ejemplo
        ``LinearSVC``). En ese caso se devuelve ``None`` en lugar de fallar.
        """
        if not self._supports_probabilities or not self._classes:
            return None

        try:
            raw = np.atleast_2d(self._model.predict_proba(frame))
        except Exception as exc:  # pragma: no cover - salvaguarda
            logger.warning("El modelo expone predict_proba pero fallo al ejecutarse: %s", exc)
            return None

        return {
            class_name: round(float(probability), 6)
            for class_name, probability in zip(self._classes, raw[0], strict=True)
        }

    @staticmethod
    def _resolve_feature_names(model: Any) -> tuple[str, ...]:
        """Obtiene los nombres de las variables en el orden que espera el modelo.

        Prioridad:
            1. ``feature_names_in_`` (existe si el modelo/pipeline se ajusto con
               un ``DataFrame``; es lo ideal y lo que produce el cuaderno 02).
            2. ``n_features_in_`` como respaldo: se generan nombres genericos
               ``feature_0..feature_n`` y se advierte en el log.
        """
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            return tuple(str(name) for name in names)

        n_features = getattr(model, "n_features_in_", None)
        if n_features:
            logger.warning(
                "El modelo no expone feature_names_in_. Se usaran nombres genericos. "
                "Recomendacion: entrene el modelo con un DataFrame con nombres de columna."
            )
            return tuple(f"feature_{i}" for i in range(int(n_features)))

        return ()

    @staticmethod
    def _resolve_classes(model: Any) -> tuple[str, ...]:
        """Obtiene las clases del modelo (``classes_``) si existen."""
        classes = getattr(model, "classes_", None)
        if classes is None:
            return ()
        return tuple(str(class_name) for class_name in classes)

    @staticmethod
    def _to_json_scalar(value: Any) -> str:
        """Convierte valores de NumPy a texto serializable como JSON."""
        item = value.item() if hasattr(value, "item") else value
        return str(item)

    def _reset(self) -> None:
        self._model = None
        self._feature_names = ()
        self._classes = ()
        self._supports_probabilities = False
        self._load_error = None
