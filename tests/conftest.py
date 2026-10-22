"""Fixtures compartidas por las pruebas.

Las pruebas **no** dependen del modelo real del proyecto: entrenan un modelo
determinista y diminuto (``random_state`` fijo) sobre el conjunto Iris de
scikit-learn y lo guardan en una carpeta temporal. Asi la suite es reproducible
y no necesita archivos privados.

Se cubren tres escenarios de modelo:
    1. ``Pipeline`` completo con ``predict_proba`` (caso normal).
    2. Modelo sin ``predict_proba`` (``LinearSVC``).
    3. Modelo cuyas variables no coinciden con el esquema HTTP.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Sequence

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from app.config import Settings
from app.main import create_app

#: Nombres de las variables que entiende el esquema HTTP de la API.
FEATURE_NAMES: tuple[str, ...] = (
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
)

#: Registro valido de ejemplo (flor Iris setosa).
VALID_PAYLOAD: dict[str, float] = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def _iris_dataframe(feature_names: Sequence[str]):
    """Devuelve (X, y) de Iris con nombres de columna personalizados."""
    data = load_iris(as_frame=True)
    features = data.frame.iloc[:, :4].copy()
    features.columns = list(feature_names)
    target = data.target.map(dict(enumerate(data.target_names)))
    return features, target


def _fit(feature_names: Sequence[str], estimator) -> Pipeline:
    """Ajusta un ``Pipeline`` (escalado + estimador) de forma determinista."""
    features, target = _iris_dataframe(feature_names)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )
    pipeline.fit(features, target)
    return pipeline


def _save(model, directory: Path) -> Path:
    """Serializa el modelo en ``directory/modelo.pkl`` y devuelve la ruta."""
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "modelo.pkl"
    joblib.dump(model, model_path)
    return model_path


@pytest.fixture(scope="session")
def deterministic_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Modelo determinista con ``predict_proba`` y variables correctas."""
    model = _fit(FEATURE_NAMES, LogisticRegression(max_iter=500, random_state=42))
    return _save(model, tmp_path_factory.mktemp("modelos") / "con_proba")


@pytest.fixture(scope="session")
def model_without_proba_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Modelo que no implementa ``predict_proba`` (``LinearSVC``)."""
    model = _fit(FEATURE_NAMES, LinearSVC(random_state=42))
    return _save(model, tmp_path_factory.mktemp("modelos") / "sin_proba")


@pytest.fixture(scope="session")
def model_with_other_features_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Modelo entrenado con nombres de variables distintos a los del esquema."""
    otras = ("x1", "x2", "x3", "x4")
    model = _fit(otras, LogisticRegression(max_iter=500, random_state=42))
    return _save(model, tmp_path_factory.mktemp("modelos") / "otras_variables")


@pytest.fixture()
def client(deterministic_model_path: Path) -> Iterator[TestClient]:
    """Cliente HTTP con un modelo determinista cargado."""
    app = create_app(
        Settings(
            model_path=deterministic_model_path,
            model_name="iris-test",
            model_version="test",
        )
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def client_without_model(tmp_path: Path) -> Iterator[TestClient]:
    """Cliente HTTP cuyo archivo de modelo no existe."""
    app = create_app(
        Settings(model_path=tmp_path / "carpeta_inexistente" / "modelo.pkl")
    )
    with TestClient(app) as test_client:
        yield test_client
