"""Pruebas automatizadas de la API.

Cubren:
    * ``GET /`` bienvenida.
    * ``GET /health`` con y sin modelo cargado.
    * ``POST /predict`` con datos validos.
    * Validaciones: campo obligatorio ausente, tipo incorrecto, campo no
      reconocido y valor fuera de rango.
    * Errores controlados: modelo no disponible (503) y variables que no
      coinciden con el modelo (422).
    * Formato JSON de la respuesta y carga unica del modelo.

Ejecutar con::

    pytest
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import FEATURE_NAMES, VALID_PAYLOAD


# --------------------------------------------------------------------------- #
# GET /
# --------------------------------------------------------------------------- #
def test_root_devuelve_la_presentacion_del_proyecto(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["project"]
    assert body["description"]
    assert body["version"]
    assert body["status"] == "ok"
    assert body["documentation"]["swagger_ui"] == "/docs"


def test_root_no_depende_del_modelo(client_without_model: TestClient) -> None:
    """El endpoint de bienvenida responde aunque el modelo no este cargado."""
    assert client_without_model.get("/").status_code == 200


# --------------------------------------------------------------------------- #
# GET /health
# --------------------------------------------------------------------------- #
def test_health_con_modelo_cargado(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "iris-test"
    assert body["model_version"] == "test"
    assert body["supports_probabilities"] is True
    assert body["features"] == list(FEATURE_NAMES)
    assert body["classes"] == ["setosa", "versicolor", "virginica"]
    assert body["detail"] is None


def test_health_sin_modelo_distingue_servidor_de_modelo(client_without_model: TestClient) -> None:
    """El servidor responde (200) pero informa que el modelo no esta disponible."""
    response = client_without_model.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["features"] == []
    assert body["detail"] == "Archivo del modelo no encontrado."


# --------------------------------------------------------------------------- #
# POST /predict - camino feliz
# --------------------------------------------------------------------------- #
def test_predict_devuelve_una_clase_valida(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"setosa", "versicolor", "virginica"}


def test_predict_devuelve_probabilidades_que_suman_uno(client: TestClient) -> None:
    body = client.post("/predict", json=VALID_PAYLOAD).json()

    probabilities = body["probabilities"]
    assert set(probabilities) == {"setosa", "versicolor", "virginica"}
    assert all(isinstance(value, float) for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_la_clase_predicha_coincide_con_la_de_mayor_probabilidad(
    client: TestClient,
) -> None:
    body = client.post("/predict", json=VALID_PAYLOAD).json()

    probabilidades = body["probabilities"]
    mas_probable = max(probabilidades, key=probabilidades.get)
    assert body["prediction"] == mas_probable


def test_predict_es_determinista(client: TestClient) -> None:
    """Con el mismo registro, la prediccion debe ser siempre la misma."""
    primera = client.post("/predict", json=VALID_PAYLOAD).json()
    segunda = client.post("/predict", json=VALID_PAYLOAD).json()

    assert primera == segunda


def test_el_orden_de_las_variables_no_altera_la_prediccion(client: TestClient) -> None:
    """El adaptador reconstruye las columnas en el orden del entrenamiento."""
    en_orden = client.post("/predict", json=VALID_PAYLOAD).json()
    en_desorden = client.post(
        "/predict",
        json={
            "petal_width": VALID_PAYLOAD["petal_width"],
            "sepal_width": VALID_PAYLOAD["sepal_width"],
            "petal_length": VALID_PAYLOAD["petal_length"],
            "sepal_length": VALID_PAYLOAD["sepal_length"],
        },
    ).json()

    assert en_orden == en_desorden


def test_formato_json_de_la_respuesta(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert set(body) == {"prediction", "probabilities", "model_name", "model_version"}
    assert isinstance(body["prediction"], str)
    assert isinstance(body["model_name"], str)
    assert isinstance(body["model_version"], str)
    assert isinstance(body["probabilities"], dict)


def test_predict_sin_predict_proba_devuelve_probabilities_null(
    model_without_proba_path: Path,
) -> None:
    """No todos los modelos admiten predict_proba; la API no debe fallar."""
    app = create_app(Settings(model_path=model_without_proba_path))

    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"setosa", "versicolor", "virginica"}
    assert body["probabilities"] is None


# --------------------------------------------------------------------------- #
# POST /predict - validaciones de entrada (422 de Pydantic/FastAPI)
# --------------------------------------------------------------------------- #
def test_predict_sin_campo_obligatorio(client: TestClient) -> None:
    payload = dict(VALID_PAYLOAD)
    del payload["petal_width"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_predict_con_tipo_de_dato_incorrecto(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "sepal_length": "cinco punto uno"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_predict_con_campo_no_reconocido(client: TestClient) -> None:
    payload = {**VALID_PAYLOAD, "color": "morado"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("sepal_length", -1.0),
        ("sepal_width", 99.0),
        ("petal_length", 0.0),
        ("petal_width", 100.0),
    ],
)
def test_predict_con_valor_fuera_de_rango(client: TestClient, campo: str, valor: float) -> None:
    payload = {**VALID_PAYLOAD, campo: valor}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


# --------------------------------------------------------------------------- #
# POST /predict - errores controlados del dominio
# --------------------------------------------------------------------------- #
def test_predict_sin_modelo_devuelve_503(client_without_model: TestClient) -> None:
    response = client_without_model.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "ModelNotAvailableError"
    assert "no esta disponible" in body["detail"]


def test_los_errores_no_filtran_rutas_internas(client_without_model: TestClient) -> None:
    mensajes = [
        client_without_model.post("/predict", json=VALID_PAYLOAD).text,
        client_without_model.get("/health").text,
    ]

    for mensaje in mensajes:
        assert "carpeta_inexistente" not in mensaje
        assert "Traceback" not in mensaje


def test_predict_con_variables_que_no_coinciden_con_el_modelo(
    model_with_other_features_path: Path,
) -> None:
    """El modelo espera x1..x4, pero el esquema HTTP envia nombres de Iris."""
    app = create_app(Settings(model_path=model_with_other_features_path))

    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "InvalidFeatureSetError"
    assert "faltan: x1, x2, x3, x4" in body["detail"]


# --------------------------------------------------------------------------- #
# Requisito: el modelo se carga UNA sola vez
# --------------------------------------------------------------------------- #
def test_el_modelo_se_carga_una_sola_vez(
    deterministic_model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cargas: list[str] = []
    joblib_load_original = joblib.load

    def joblib_load_espia(*args, **kwargs):
        cargas.append(str(args[0]))
        return joblib_load_original(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", joblib_load_espia)

    app = create_app(Settings(model_path=deterministic_model_path))
    with TestClient(app) as test_client:
        test_client.get("/health")
        test_client.post("/predict", json=VALID_PAYLOAD)
        test_client.post("/predict", json=VALID_PAYLOAD)
        test_client.post("/predict", json=VALID_PAYLOAD)

    assert len(cargas) == 1, "El modelo debe leerse del disco solo durante el arranque"


# --------------------------------------------------------------------------- #
# Documentacion interactiva
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ruta", ["/docs", "/redoc", "/openapi.json"])
def test_documentacion_interactiva_disponible(client: TestClient, ruta: str) -> None:
    assert client.get(ruta).status_code == 200


def test_openapi_documenta_los_endpoints(client: TestClient) -> None:
    esquema = client.get("/openapi.json").json()

    assert set(esquema["paths"]) == {"/", "/health", "/predict"}
