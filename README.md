# API del modelo de Machine Learning (FastAPI + `.pkl` + arquitectura hexagonal)

Proyecto base, didactico y reutilizable para **publicar un modelo de Machine Learning ya entrenado** como una **API REST**.

El modelo se entrena **antes** (en un cuaderno de Jupyter) y se guarda en un archivo `.pkl`. La API se encarga unicamente de:

```
recibir datos -> validarlos -> cargar el modelo -> ejecutar la inferencia -> devolver JSON
```

La aplicacion usa **arquitectura hexagonal (puertos y adaptadores)**, de modo que el nucleo del negocio no depende de FastAPI ni de scikit-learn.

---

## Tabla de contenidos

1. [Descripcion y objetivo](#1-descripcion-y-objetivo)
2. [Enfoque pedagogico: el ciclo de vida completo](#2-enfoque-pedagogico-el-ciclo-de-vida-completo)
3. [Arquitectura de la solucion](#3-arquitectura-de-la-solucion)
4. [Estructura de carpetas](#4-estructura-de-carpetas)
5. [Explicacion de cada archivo](#5-explicacion-de-cada-archivo)
6. [Requisitos previos](#6-requisitos-previos)
7. [Instalacion](#7-instalacion)
8. [Configuracion del modelo `.pkl`](#8-configuracion-del-modelo-pkl)
9. [Ejecucion local paso a paso](#9-ejecucion-local-paso-a-paso)
10. [Endpoints disponibles](#10-endpoints-disponibles)
11. [Ejemplos de solicitudes y respuestas](#11-ejemplos-de-solicitudes-y-respuestas)
12. [Probar `POST /predict` desde Swagger UI](#12-probar-post-predict-desde-swagger-ui)
13. [Cuadernos de Jupyter](#13-cuadernos-de-jupyter)
14. [Ejecucion de las pruebas](#14-ejecucion-de-las-pruebas)
15. [Despliegue en un servicio de hosting](#15-despliegue-en-un-servicio-de-hosting)
16. [Problemas frecuentes y soluciones](#16-problemas-frecuentes-y-soluciones)
17. [Consideraciones de seguridad](#17-consideraciones-de-seguridad)
18. [Limitaciones conocidas](#18-limitaciones-conocidas)
19. [Lista de verificacion](#19-lista-de-verificacion)

---

## 1. Descripcion y objetivo

**Objetivo general:** construir una API que cargue un modelo de Machine Learning desde un archivo `.pkl`, reciba datos por HTTP, aplique las transformaciones necesarias, genere predicciones y las devuelva en formato JSON.

**El modelo NO se entrena durante las solicitudes.** Se carga **una sola vez** al arrancar el servidor.

**Modelo de ejemplo incluido en este repositorio**

| Elemento | Valor |
|---|---|
| Conjunto de datos | **Iris** de `scikit-learn` (base de datos de prueba, 150 registros, 4 variables) |
| Problema | Clasificacion multiclase (3 especies) |
| Modelo | `Pipeline(StandardScaler -> LogisticRegression)` |
| Archivo | `models/modelo.pkl` (generado por el cuaderno 02, aprox. 2 KB) |
| Exactitud en prueba | **0.9333** (30 registros de prueba, `random_state=42`) |
| Version de `predict_proba()` | Soportada (devuelve probabilidad por clase) |

> Los nombres, tipos y rangos de las variables del modelo de ejemplo son los de Iris.
> **Si entrenas con otro conjunto de datos debes adaptar el esquema Pydantic** (ver [seccion 8](#8-configuracion-del-modelo-pkl) y `app/adapters/inbound/http/schemas.py`).

**Tecnologias:** Python 3.11+ · FastAPI · Uvicorn · scikit-learn · joblib · Pydantic · pandas · NumPy · Pytest · Git/GitHub · `uv` (opcional).

---

## 2. Enfoque pedagogico: el ciclo de vida completo

Conviene distinguir seis acciones que suelen confundirse:

| Accion | Pregunta que responde | Donde ocurre en este proyecto |
|---|---|---|
| **Entrenar** | De los datos, cual es el patron? | `notebooks/02_entrenamiento_modelo.ipynb` |
| **Guardar** | Como conservo lo aprendido? | `joblib.dump(pipeline, "models/modelo.pkl")` |
| **Cargar** | Como recupero el modelo en otro proceso? | `JoblibModelGateway.load()` |
| **Exponer** | Como ofrezco el modelo a otras aplicaciones? | `POST /predict` (adaptador HTTP) |
| **Ejecutar** | Que proceso mantiene la API viva? | `uvicorn app.main:app` |
| **Consumir** | Quien usa la API? | Swagger UI, un notebook, un frontend, `curl` |

**Diagrama de arquitectura**

```text
                        FASE DE ENTRENAMIENTO (offline, una sola vez)
  Datos  ->  Preparacion  ->  Entrenamiento  ->  Evaluacion  ->  Modelo .pkl
                                                                     |
                                                                     | se copia a models/
                                                                     v
                        FASE DE SERVICIO (online, en cada solicitud)
  Solicitud HTTP  ->  FastAPI  ->  Validacion (Pydantic)  ->  Carga unica del .pkl
                                                                     |
                                                                     v
                                                     model.predict() / predict_proba()
                                                                     |
                                                                     v
                                                        Respuesta JSON  ->  Cliente
```

Puntos clave para el estudiante:

1. El entrenamiento es **offline** y costoso; la inferencia es **online** y barata.
2. El modelo se carga al **arrancar** el servidor, no en cada solicitud (cargarlo en cada `POST` seria lentisimo).
3. La API **no reentrena**: si el rendimiento decae, se reentrena aparte y se reemplaza el `.pkl`.
4. Uvicorn es el **servidor ASGI** que ejecuta la aplicacion FastAPI; no es una libreria de Machine Learning.
5. La API es el **punto de integracion**: cualquier lenguaje o aplicacion puede consumirla por HTTP.

---

## 3. Arquitectura de la solucion

### 3.1 Por que hexagonal

En una arquitectura hexagonal el **dominio** (las reglas del negocio) esta en el centro y la tecnologia queda en los bordes:

```text
                    +-------------------------------------------+
   Adaptador de     |              DOMINIO                      |    Adaptador de
   entrada (HTTP)   |  entidades | puertos | errores           |    salida (.pkl)
   +-------------+  |                                           |  +-----------------+
   |  rutas      |->|  PredictUseCase  ->  ModelGatewayPort      |->| JoblibModel       |
   |  schemas    |  |  GetStatusUseCase                         |  | Gateway           |
   |  dependencias| |                                           |  | (joblib/sklearn) |
   +-------------+  +-------------------------------------------+
        FastAPI                casos de uso                            scikit-learn
```

Beneficios didacticos y practicos:

- **Testeable:** los casos de uso se prueban sin levantar FastAPI y sin un `.pkl` real (el puerto se puede sustituir).
- **Intercambiable:** cambiar FastAPI por Flask, o scikit-learn por otro motor (ONNX, XGBoost), solo afecta a un adaptador.
- **Legible:** la capa de aplicacion describe *que* hace la API; los adaptadores describen *como*.

### 3.2 Flujo de una prediccion

```text
POST /predict
   |
   v
[rutas.py]  endpoint FastAPI  ------------------> depende de
   |
   v
[dependencies.py]  inyecta el caso de uso desde app.state
   |
   v
[PredictUseCase.execute()]
   |-- 1. gateway.info.loaded?  .. si no -> ModelNotAvailableError (503)
   |-- 2. valida que las variables coincidan con el modelo
   |        .. si no -> InvalidFeatureSetError (422)
   v
[JoblibModelGateway.predict()]
   |-- 3. reconstruye el DataFrame con los nombres y el ORDEN del entrenamiento
   |-- 4. model.predict()  y  model.predict_proba() (si existe)
   |-- 5. convierte numpy -> tipos JSON
   v
[Prediction]  ->  [PredictResponse]  ->  JSON
```

### 3.3 Traduccion de errores de dominio a HTTP

| Error de dominio | HTTP | Significado |
|---|---|---|
| `InvalidFeatureSetError` | 422 | Las variables enviadas no son las del modelo |
| `ModelNotAvailableError` | 503 | El servidor responde, pero el modelo no esta cargado |
| `InferenceError` | 500 | El modelo fallo al predecir |
| `RequestValidationError` (Pydantic/FastAPI) | 422 | Falta un campo, tipo incorrecto, valor fuera de rango |

---

## 4. Estructura de carpetas

```text
Modelo_guia_fastapi/
│
├── app/                                    # Codigo de la API (hexagono)
│   ├── __init__.py
│   ├── config.py                           # Configuracion por variables de entorno
│   ├── main.py                             # Punto de entrada + composicion de dependencias
│   │
│   ├── domain/                             # NUCLEO (sin FastAPI, sin sklearn)
│   │   ├── __init__.py
│   │   ├── entities.py                     # ModelInfo, Prediction
│   │   ├── ports.py                        # ModelGatewayPort (interfaz)
│   │   └── exceptions.py                   # Errores del negocio
│   │
│   ├── application/                        # CASOS DE USO
│   │   ├── __init__.py
│   │   ├── predict_use_case.py             # Validar entrada + predecir
│   │   └── status_use_case.py              # Consultar estado del modelo
│   │
│   └── adapters/                           # ADAPTADORES (tecnologia)
│       ├── __init__.py
│       ├── inbound/                        # driving: entran solicitudes
│       │   └── http/
│       │       ├── __init__.py
│       │       ├── routes.py               # GET /, GET /health, POST /predict
│       │       ├── schemas.py              # Esquemas Pydantic (entrada/salida)
│       │       └── dependencies.py         # Inyeccion de dependencias
│       └── outbound/                       # driven: sale hacia el modelo
│           ├── __init__.py
│           └── joblib_model_gateway.py     # Carga el .pkl y ejecuta la inferencia
│
├── models/
│   ├── .gitkeep
│   └── modelo.pkl                          # Modelo entrenado (generado por el cuaderno 02)
│
├── notebooks/                              # Cuadernos didacticos
│   ├── 01_analisis_descriptivo.ipynb       # Analisis descriptivo de los datos (EDA)
│   └── 02_entrenamiento_modelo.ipynb       # Entrenamiento, evaluacion y guardado del .pkl
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                         # Fixtures: modelo determinista de prueba
│   └── test_api.py                         # Pruebas de los endpoints
│
├── .gitignore
├── pyproject.toml                          # Metadatos, dependencias y config de pytest
├── requirements.txt                        # Dependencias de ejecucion/despliegue
└── README.md
```

> **Equivalencia con la estructura "clasica"** que suele pedirse en clase:
> `app/schemas.py` -> `app/adapters/inbound/http/schemas.py`;
> `app/model_service.py` -> `app/adapters/outbound/joblib_model_gateway.py`;
> `app/main.py` y `app/config.py` se mantienen en la raiz de `app/`.

---

## 5. Explicacion de cada archivo

| Archivo | Que hace | Por que esta ahi |
|---|---|---|
| `app/config.py` | Define `Settings` (ruta del `.pkl`, nombre, version, host, puerto) y lo lee de variables de entorno. | Un unico lugar de configuracion, inyectable y sin dependencias de framework. |
| `app/main.py` | Crea la app con `create_app()`, carga el modelo en el evento `lifespan`, guarda los casos de uso en `app.state` y traduce errores de dominio a HTTP. | Es el **composition root**: une todas las piezas. Expone `app` para Uvicorn. |
| `app/domain/entities.py` | `ModelInfo` (estado del modelo) y `Prediction` (etiqueta + probabilidades). | Vocabulario del negocio, inmutable y sin numpy. |
| `app/domain/ports.py` | `ModelGatewayPort`: contrato `load()`, `info`, `predict()`. | Define *que* se necesita del exterior sin decir *como* se implementa. |
| `app/domain/exceptions.py` | `ModelNotAvailableError`, `InvalidFeatureSetError`, `InferenceError`. | Errores con mensaje publico y detalle privado. |
| `app/application/predict_use_case.py` | Verifica que el modelo este cargado, valida las variables y delega la prediccion. | Orquesta el caso de uso sin conocer HTTP ni scikit-learn. |
| `app/application/status_use_case.py` | Devuelve los metadatos del modelo. | Lo usa `/health`. |
| `app/adapters/inbound/http/routes.py` | Los tres endpoints. | Adaptador de entrada: traduce HTTP a casos de uso. |
| `app/adapters/inbound/http/schemas.py` | Esquemas Pydantic de entrada/salida con validacion de tipos y rangos. | Contrato explicito y documentacion automatica. |
| `app/adapters/inbound/http/dependencies.py` | Recupera configuracion y casos de uso desde `app.state`. | Evita instanciar el modelo por solicitud. |
| `app/adapters/outbound/joblib_model_gateway.py` | Carga el `.pkl` con `joblib`, detecta `feature_names_in_`/`classes_`/`predict_proba` y ejecuta la inferencia. | Unico lugar que conoce scikit-learn. |
| `models/modelo.pkl` | Modelo entrenado (pipeline completo). | Artefacto que la API consume. |
| `notebooks/01_...ipynb` | Analisis descriptivo de Iris. | Justifica las decisiones de preprocesamiento. |
| `notebooks/02_...ipynb` | Entrena, evalua, guarda y **verifica** el `.pkl`. | Produce el artefacto desplegable. |
| `tests/conftest.py` | Fixtures: entrena un modelo diminuto y determinista en una carpeta temporal. | Hace las pruebas reproducibles y sin dependencias externas. |
| `tests/test_api.py` | 26 pruebas de los endpoints y de las validaciones. | Verifica el comportamiento observable de la API. |
| `pyproject.toml` | Dependencias, extras `dev`/`notebooks` y configuracion de `pytest`. | Empaquetado e instalacion reproducible. |
| `requirements.txt` | Dependencias de ejecucion (produccion). | Instalacion simple en el hosting. |

---

## 6. Requisitos previos

- **Python 3.11 o superior** (probado con 3.11.14 y 3.12.4).

```powershell
python --version
```

- **`git`** (para control de versiones y despliegue).

```powershell
git --version
```

- **`uv`** (opcional, recomendado por su velocidad). Si no lo tienes:

```powershell
pip install uv
```

- Sin conocimientos previos de Docker: no se necesita.

---

## 7. Instalacion

### Opcion A - Con `uv` (recomendada)

```powershell
cd d:\GitHub\Modelo_guia_fastapi
uv venv
uv pip install -e ".[dev,notebooks]"
```

`uv` crea `.venv` y resuelve las dependencias del `pyproject.toml`, incluidos los extras:

- `dev`: `pytest`, `httpx` (para `TestClient`).
- `notebooks`: `jupyterlab`, `matplotlib`.

### Opcion B - Sin `uv` (venv clasico)

```powershell
cd d:\GitHub\Modelo_guia_fastapi
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -e ".[dev,notebooks]"
```

### Opcion C - Solo produccion / despliegue

```powershell
pip install -r requirements.txt
```

> `requirements.txt` contiene unicamente lo necesario para ejecutar la API. `pytest` y `jupyterlab` no se instalan en el servidor.

### Registrar el kernel del proyecto para los cuadernos (opcional)

Si en tu maquina hay varias instalaciones de Python, registra el kernel del entorno virtual para que los cuadernos usen **exactamente** las mismas versiones que la API:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name api-modelo-ml --display-name "API Modelo ML (venv)"
```

---

## 8. Configuracion del modelo `.pkl`

### 8.1 Escenario 2 (recomendado): el `.pkl` contiene un `Pipeline` completo

Es el caso de este proyecto. El pipeline incluye `StandardScaler` y el clasificador, ya ajustados. La API lo usa directamente:

```python
pipeline.predict(df)   # escala con los parametros del entrenamiento y luego clasifica
```

**Ventajas:** las transformaciones quedan congeladas dentro del artefacto; no hay riesgo de olvidar un paso.

### 8.2 Escenario 1: el `.pkl` contiene solo el estimador

Si el archivo tiene unicamente el modelo final, **debes reproducir en la API las mismas transformaciones del entrenamiento**:

1. Mismo orden y nombres de columnas que en el entrenamiento.
2. Mismas decisiones de limpieza (nulos, outliers, tipos).
3. Mismas codificaciones de variables categoricas (variables dummy / `OneHotEncoder`).
4. Misma normalizacion o estandarizacion, usando **los parametros del entrenamiento**.

Lo mas seguro es guardar tambien los objetos ajustados:

```python
# En el cuaderno de entrenamiento
joblib.dump({"modelo": modelo, "escalador": escalador, "columnas": list(X.columns)}, "models/modelo.pkl")
```

```python
# En el adaptador de salida
artefacto = joblib.load(path)
X = pd.DataFrame([registro], columns=artefacto["columnas"])
X_esc = artefacto["escalador"].transform(X)      # transform, NUNCA fit_transform
prediccion = artefacto["modelo"].predict(X_esc)
```

### 8.3 `ColumnTransformer`, codificadores y escaladores: por que reutilizar y no reajustar

Un `StandardScaler` que hizo `fit` durante el entrenamiento guarda `mean_` y `scale_`. Un `OneHotEncoder` guarda las categorias vistas. Un `ColumnTransformer` guarda que columnas van a cada rama.

Si **reajustas** con los datos de la solicitud:

- La media y la desviacion se calcularian sobre **una sola fila**: el resultado siempre seria 0, la escala se perderia y las predicciones serian basura.
- Una categoria no vista cambiaria el numero de columnas respecto al entrenamiento (`ValueError: feature names/order mismatch`).
- Cada solicitud daria una prediccion distinta para el mismo dato: el modelo dejaria de ser reproducible.

Por eso, **siempre** `transform()` con los objetos ya ajustados. Al guardar el `Pipeline` completo esto se cumple automaticamente.

### 8.4 Donde colocar el archivo

```text
models/modelo.pkl
```

Casos de uso:

- **Modelo de ejemplo de este repositorio:** ya esta incluido. No hagas nada.
- **Modelo entrenado por ti:** ejecuta `notebooks/02_entrenamiento_modelo.ipynb`; el archivo se escribe en `models/modelo.pkl`.
- **Modelo entrenado por otra persona:** copia el archivo a `models/` conservando el nombre `modelo.pkl`.
- **Modelo en otra ruta o con otro nombre:** no cambies el codigo, usa la variable de entorno:

```powershell
$env:MODEL_PATH = "D:\ruta\a\mi_modelo.pkl"
```

Si el archivo no existe, la API **arranca igual**, pero:

- `GET /health` responde `200` con `"model_loaded": false` y `"status": "degraded"`.
- `POST /predict` responde **503** con un mensaje claro (nunca una traza interna).

### 8.5 Compatibilidad de versiones

Un `.pkl` de scikit-learn depende de las versiones con que se guardo (`numpy`, `scipy`, `sklearn`). Guarda el registro y replicarlo en el servidor:

| Libreria | Version usada para entrenar `models/modelo.pkl` |
|---|---|
| Python | 3.11.14 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| pandas | 3.0.6 |
| numpy | 2.4.6 |

> Si actualizas `scikit-learn`, **reentrena y vuelve a guardar** el modelo. Si solo cambias de maquina, instala las mismas versiones.

---

## 9. Ejecucion local paso a paso

### Paso 1 - Verificar Python

```powershell
python --version
```

### Paso 2 - Crear y activar el entorno virtual

Con `uv`:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
```

Con `venv` clasico:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Paso 3 - Instalar las dependencias

```powershell
uv pip install -e ".[dev,notebooks]"
```

### Paso 4 - Copiar el modelo entrenado

```text
models/modelo.pkl
```

Revisa la [seccion 8.4](#84-donde-colocar-el-archivo). En este repositorio el archivo ya viene generado.

### Paso 5 - Ejecutar la API

Desde la **raiz del proyecto**:

```powershell
uvicorn app.main:app --reload
```

Sin activar el entorno:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

> `--reload` reinicia el servidor al detectar cambios en el codigo: **comodo en desarrollo, no lo uses en produccion** (consume recursos y no es seguro). Para produccion:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Al arrancar veras en los logs algo similar a:

```text
Cargando el modelo una unica vez (ruta configurada: ...\models\modelo.pkl)
Modelo 'iris-classifier' v1.0.0 cargado desde ... | variables=4 | clases=3 | predict_proba=True
```

### Paso 6 - Abrir Swagger UI

```text
http://127.0.0.1:8000/docs
```

### Paso 7 - Enviar una solicitud de prueba

Desde Swagger UI ([seccion 12](#12-probar-post-predict-desde-swagger-ui)) o desde PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType "application/json" `
  -Body '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}' |
  ConvertTo-Json
```

### Paso 8 - Interpretar la prediccion

```json
{
  "prediction": "setosa",
  "probabilities": { "setosa": 0.980813, "versicolor": 0.019187, "virginica": 0.0 },
  "model_name": "iris-classifier",
  "model_version": "1.0.0"
}
```

- `prediction`: la clase mas probable (equivalente al mayor valor de `probabilities`).
- `probabilities`: probabilidad de cada clase, suma 1.0. Es `null` si el modelo no implementa `predict_proba()`.
- Al interpretar: una probabilidad cercana a 0.5 entre dos clases indica **frontera de decision**, no certeza.

### Paso 9 - Ejecutar las pruebas automatizadas

```powershell
pytest
```

---

## 10. Endpoints disponibles

| Metodo | Ruta | Proposito | Codigos de respuesta |
|---|---|---|---|
| `GET` | `/` | Presentacion del proyecto y su estado general | `200` |
| `GET` | `/health` | Estado del servidor **y** del modelo | `200` siempre que el servidor responda |
| `POST` | `/predict` | Generar una prediccion | `200`, `422`, `503`, `500` |
| `GET` | `/docs` | Documentacion interactiva (Swagger UI) | `200` |
| `GET` | `/redoc` | Documentacion alternativa (ReDoc) | `200` |
| `GET` | `/openapi.json` | Esquema OpenAPI generado | `200` |

### Diferencia entre servidor activo y modelo disponible

`/health` separa dos conceptos que suelen confundirse:

- **El servidor responde** (`GET /health` devuelve `200`): el proceso Uvicorn esta vivo.
- **El modelo esta disponible** (`"model_loaded": true`): existe en memoria y `/predict` funciona.

Un despliegue saludable necesita **ambos**. Si el modelo no se cargo, `/health` responde `200` con `"status": "degraded"` y el sistema de monitoreo del hosting puede detectarlo.

---

## 11. Ejemplos de solicitudes y respuestas

### `GET /`

```bash
curl http://127.0.0.1:8000/
```

```json
{
  "project": "API del modelo de Machine Learning",
  "description": "API REST construida con FastAPI que carga un modelo de Machine Learning previamente entrenado (archivo .pkl) y expone predicciones por HTTP. Proyecto base con arquitectura hexagonal para fines didacticos.",
  "version": "0.1.0",
  "status": "ok",
  "documentation": { "swagger_ui": "/docs", "redoc": "/redoc", "openapi": "/openapi.json" }
}
```

### `GET /health` (modelo cargado)

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "iris-classifier",
  "model_version": "1.0.0",
  "supports_probabilities": true,
  "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "classes": ["setosa", "versicolor", "virginica"],
  "detail": null
}
```

### `GET /health` (modelo no disponible)

```json
{
  "status": "degraded",
  "model_loaded": false,
  "model_name": "iris-classifier",
  "model_version": "1.0.0",
  "supports_probabilities": false,
  "features": [],
  "classes": [],
  "detail": "Archivo del modelo no encontrado."
}
```

### `POST /predict` - solicitud valida (clase `setosa`)

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Respuesta `200`:

```json
{
  "prediction": "setosa",
  "probabilities": { "setosa": 0.980813, "versicolor": 0.019187, "virginica": 0.0 },
  "model_name": "iris-classifier",
  "model_version": "1.0.0"
}
```

### `POST /predict` - otra clase (`virginica`)

```json
{ "sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3 }
```

```json
{
  "prediction": "virginica",
  "probabilities": { "setosa": 0.00006, "versicolor": 0.043472, "virginica": 0.956468 },
  "model_name": "iris-classifier",
  "model_version": "1.0.0"
}
```

### `POST /predict` - campo obligatorio ausente (`422`)

Solicitud:

```json
{ "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4 }
```

Respuesta (formato de validacion de FastAPI, `detail` es una lista):

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "petal_width"],
      "msg": "Field required",
      "input": { "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4 }
    }
  ]
}
```

### `POST /predict` - tipo de dato incorrecto (`422`)

```json
{ "sepal_length": "cinco", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2 }
```

### `POST /predict` - modelo no disponible (`503`)

```json
{
  "detail": "El modelo no esta disponible en este momento. Consulte GET /health para mas informacion.",
  "error": "ModelNotAvailableError"
}
```

### `POST /predict` - variables que no coinciden con el modelo (`422`)

Si el `.pkl` espera otras variables (por ejemplo `x1..x4`):

```json
{
  "detail": "Las variables enviadas no coinciden con las que espera el modelo (faltan: x1, x2, x3, x4; no reconocidas: petal_length, petal_width, sepal_length, sepal_width).",
  "error": "InvalidFeatureSetError"
}
```

### Si el modelo no implementa `predict_proba()`

```json
{
  "prediction": "setosa",
  "probabilities": null,
  "model_name": "iris-classifier",
  "model_version": "1.0.0"
}
```

La API **no falla**: comprueba la capacidad con `hasattr(modelo, "predict_proba")` y devuelve `null`. Modelos como `LinearSVC` no la implementan; si necesitas probabilidades, usa `SVC(probability=True)`, `CalibratedClassifierCV` o `LogisticRegression`.

---

## 12. Probar `POST /predict` desde Swagger UI

1. Ejecuta `uvicorn app.main:app --reload` y abre `http://127.0.0.1:8000/docs`.
2. Despliega el grupo **Modelo** y haz clic en `POST /predict`.
3. Pulsa **Try it out**.
4. Sustituye el contenido del cuadro *Request body* por un JSON valido:

   ```json
   {
     "sepal_length": 5.1,
     "sepal_width": 3.5,
     "petal_length": 1.4,
     "petal_width": 0.2
   }
   ```

   > El esquema de ejemplo ya viene precargado con estos valores.
5. Pulsa **Execute**.
6. Revisa tres zonas:
   - **Curl**: el comando equivalente para reproducirlo en la terminal.
   - **Request URL**: `POST http://127.0.0.1:8000/predict`.
   - **Server response**: el codigo (`200`) y el cuerpo JSON con `prediction` y `probabilities`.
7. Experimenta con errores: borra `petal_width` (=> `422`), envia `"sepal_length": "alto"` (=> `422`) o un valor como `-3` (=> `422`).

Los esquemas de entrada y salida tambien aparecen al final de la pagina, en **Schemas**, y son la documentacion que consumira otro equipo de desarrollo.

---

## 13. Cuadernos de Jupyter

```powershell
jupyter lab notebooks
```

| Cuaderno | Contenido |
|---|---|
| `01_analisis_descriptivo.ipynb` | Carga de Iris, tipos, nulos, duplicados, balance de clases, estadistica descriptiva, correlaciones, histogramas, boxplots, dispersion y conclusiones. Define el **contrato de variables**. |
| `02_entrenamiento_modelo.ipynb` | Division entrenamiento/prueba, `Pipeline(StandardScaler + LogisticRegression)`, evaluacion (`accuracy` = 0.9333, `classification_report`, matriz de confusion), ejemplo de inferencia, guardado con `joblib`, verificacion de la carga y registro de versiones. |

Ambos cuadernos **ya estan ejecutados y con salidas guardadas**, de modo que pueden leerse sin volver a ejecutarlos. El modelo `models/modelo.pkl` fue generado ejecutando el cuaderno 02.

> Si ejecutas el cuaderno 02, **sobrescribira** `models/modelo.pkl`.

---

## 14. Ejecucion de las pruebas

```powershell
pytest                 # suite completa
pytest -v              # con el nombre de cada prueba
pytest tests/test_api.py::test_predict_devuelve_una_clase_valida
```

**Resultado verificado en este repositorio:** `26 passed` (aprox. 2 segundos).

Cobertura de la suite:

| Requisito | Prueba |
|---|---|
| Respuesta de `GET /` | `test_root_devuelve_la_presentacion_del_proyecto`, `test_root_no_depende_del_modelo` |
| Estado de `GET /health` | `test_health_con_modelo_cargado`, `test_health_sin_modelo_distingue_servidor_de_modelo` |
| Prediccion valida | `test_predict_devuelve_una_clase_valida`, `test_predict_es_determinista` |
| Campo obligatorio ausente | `test_predict_sin_campo_obligatorio` |
| Tipos de datos incorrectos | `test_predict_con_tipo_de_dato_incorrecto` |
| Campo no reconocido / fuera de rango | `test_predict_con_campo_no_reconocido`, `test_predict_con_valor_fuera_de_rango` |
| Modelo no disponible | `test_predict_sin_modelo_devuelve_503`, `test_los_errores_no_filtran_rutas_internas` |
| Variables que no coinciden con el modelo | `test_predict_con_variables_que_no_coinciden_con_el_modelo` |
| Modelo sin `predict_proba` | `test_predict_sin_predict_proba_devuelve_probabilities_null` |
| Formato de la respuesta JSON | `test_formato_json_de_la_respuesta`, `test_predict_devuelve_probabilidades_que_suman_uno` |
| Orden de las variables irrelevante | `test_el_orden_de_las_variables_no_altera_la_prediccion` |
| Carga unica del modelo | `test_el_modelo_se_carga_una_sola_vez` |
| Documentacion interactiva | `test_documentacion_interactiva_disponible`, `test_openapi_documenta_los_endpoints` |

**Las pruebas no dependen del `.pkl` real.** `tests/conftest.py` entrena en una carpeta temporal un modelo diminuto y **determinista** (`random_state=42`) y ademas construye, cuando hace falta, un modelo sin `predict_proba` y otro con nombres de variables distintos.

---

## 15. Despliegue en un servicio de hosting

Esta seccion es **generica y adaptable**: no asume un proveedor concreto. Verifica siempre en la documentacion del proveedor elegido si soporta contenedores, procesos ASGI y volumenes persistentes.

### 15.1 Comando de inicio con puerto dinamico

El proveedor casi siempre inyecta el puerto en la variable `PORT`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

- `--host 0.0.0.0` es obligatorio para aceptar trafico externo.
- `${PORT:-8000}` usa el valor por defecto `8000` si la variable no esta definida.
- **No uses `--reload`** en produccion.

Archivo `Procfile` (si el proveedor lo soporta):

```text
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

En Windows no existe `${VAR:-default}`; si el hosting es Windows, define la variable `PORT` y usa `--port %PORT%`.

### 15.2 Instalacion reproducible de dependencias

**Comando de compilacion / build:**

```bash
pip install -r requirements.txt
```

Recomendaciones:

- Fija versiones exactas (`scikit-learn==1.9.1`, `numpy==2.4.6`, ...) para reproducir el entorno de entrenamiento. Con `uv` puedes generar el archivo congelado:

  ```bash
  uv pip freeze > requirements.lock.txt
  ```

- Fija la version de Python en el proveedor (por ejemplo 3.11), la misma con la que se entreno el modelo.

### 15.3 Conectar el repositorio de GitHub

```powershell
git init
git add .
git commit -m "Proyecto base de API para modelo de ML"
git branch -M main
git remote add origin https://github.com/<usuario>/<repositorio>.git
git push -u origin main
```

Luego, en el panel del proveedor: **New project -> Deploy from GitHub repository**, selecciona el repositorio y la rama (`main`). Configura el comando de inicio y, si es necesario, el comando de compilacion.

> Revisa `.gitignore`: los modelos grandes se excluyen por defecto. El `.pkl` de ejemplo (2 KB) si se versiona. Si tu modelo pesa cientos de MB, no lo subas a GitHub: usalo desde un almacenamiento externo (S3, GCS, volumen del proveedor) y descargalo al arrancar, o monta un disco persistente y define `MODEL_PATH`.

### 15.4 Variables de entorno

| Variable | Por defecto | Descripcion |
|---|---|---|
| `MODEL_PATH` | `models/modelo.pkl` | Ruta del modelo entrenado |
| `MODEL_NAME` | `iris-classifier` | Nombre logico que se reporta en las respuestas |
| `MODEL_VERSION` | `1.0.0` | Version del modelo |
| `APP_NAME` | API del modelo de Machine Learning | Titulo en la documentacion |
| `APP_VERSION` | `0.1.0` | Version de la API |
| `HOST` | `0.0.0.0` | Interfaz de escucha (se usa en `python -m app.main`) |
| `PORT` | `8000` | Puerto de escucha |

### 15.5 Verificacion despues del despliegue

1. **Registros del servidor (logs):** busca las dos lineas del arranque:

   ```text
   Cargando el modelo una unica vez (ruta configurada: ...)
   Modelo 'iris-classifier' v1.0.0 cargado desde ... | variables=4 | clases=3 | predict_proba=True
   ```

   Si aparece `La API arrancara en modo degradado`, el `.pkl` no se encontro o no pudo deserializarse.

2. **`GET /health`:**

   ```bash
   curl https://<tu-dominio>/health
   ```

   Debe responder `"status": "ok"` y `"model_loaded": true`.

3. **`POST /predict`:**

   ```bash
   curl -X POST https://<tu-dominio>/predict \
     -H "Content-Type: application/json" \
     -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
   ```

4. Configura el **health check** del proveedor apuntando a `/health`.

5. Mide el **tiempo de respuesta** (latencia) y el uso de memoria despues de varias solicitudes: deben mantenerse estables (el modelo se carga una sola vez).

### 15.6 Tamano del modelo y memoria

- El modelo vive **en memoria** mientras el proceso esta activo. Una instancia con poca RAM puede reiniciarse por falta de memoria (`OOM`) si el `.pkl` es grande.
- Regla practica: mide el **pico** de memoria al cargar (`/health` y observa las metricas de la plataforma) y deja margen de al menos 2x.
- Los planes gratuitos suelen **suspenderse** por inactividad: la primera solicitud tarda mas porque el proceso debe arrancar y cargar el modelo.
- Si el modelo supera el limite del proveedor, considera: reducir el modelo (menos arboles, `int8`), comprimirlo, o usar un servicio de inferencia separado.

### 15.7 Seguridad, limites y validacion

- Sirve siempre por **HTTPS** (los proveedores lo ofrecen por defecto).
- **No expongas detalles internos** en las respuestas: esta API ya devuelve mensajes genericos y deja los detalles en los logs.
- **Limita el tamano** del cuerpo de la solicitud y la **tasa de peticiones** (rate limiting) en el proxy o middleware.
- Restringe **CORS** a los dominios que realmente consumen la API (`CORSMiddleware` con origenes explicitos).
- Considera una **API key** o `OAuth2` si los datos son sensibles.
- Valida siempre con **Pydantic** (ya implementado) y rechaza campos no declarados (`extra="forbid"`).
- Los `.pkl` pueden ejecutar codigo al deserializarse: **solo** carga modelos propios o de fuentes confiables.
- Registra auditoria (quien, cuando, que) si el modelo toma decisiones con impacto sobre personas.

---

## 16. Problemas frecuentes y soluciones

| Sintoma | Causa probable | Solucion |
|---|---|---|
| `El modelo no esta disponible` / `/health` con `model_loaded: false` | El archivo no esta en la ruta configurada | Coloca el `.pkl` en `models/modelo.pkl` o define `MODEL_PATH` |
| `No fue posible deserializar el archivo del modelo` | Version distinta de scikit-learn/numpy que al entrenar | Instala las mismas versiones (revisa los logs del servidor) o reentrena el modelo |
| `ModuleNotFoundError: No module named 'app'` | Ejecutas Uvicorn desde otra carpeta | Ejecuta desde la **raiz del proyecto**: `uvicorn app.main:app --reload` |
| `Address already in use` (puerto ocupado) | Ya hay un proceso en el puerto 8000 | Usa otro puerto: `uvicorn app.main:app --port 8001` |
| `Las variables enviadas no coinciden con las que espera el modelo` | El esquema Pydantic no coincide con `feature_names_in_` | Alinea los nombres en `schemas.py`; consulta `/health` para ver `features` |
| `predictions: null` en la respuesta | El modelo no implementa `predict_proba()` | Es esperado; usa un clasificador con probabilidades si las necesitas |
| `422` al enviar variables correctas | `extra="forbid"` rechaza campos desconocidos | Envia **unicamente** las variables del modelo |
| Los cuadernos usan otra version de las librerias | El kernel de Jupyter es de otra instalacion de Python | Registra el kernel del venv (seccion 7) y seleccionaolo en el cuaderno |
| `ImportError: numpy.dtype size changed` | NumPy incompatible con las librerias compiladas | Reinstala el entorno: `uv pip install -e ".[dev,notebooks]"` |
| El contenedor se reinicia en el hosting | Falta de memoria al cargar el modelo | Aumenta la memoria o reduce el tamano del modelo |

---

## 17. Consideraciones de seguridad

1. **Deserializacion segura.** `pickle`/`joblib` pueden ejecutar codigo arbitrario. Nunca cargues un `.pkl` de origen desconocido; en produccion, descarga el modelo por HTTPS desde un almacenamiento propio y verifica su integridad.
2. **Sin filtrado de informacion.** Las respuestas no incluyen rutas, trazas ni versiones internas de librerias; los detalles tecnicos solo van a los logs.
3. **Validacion estricta.** Tipos, rangos y campos cerrados con Pydantic; las variables se contrastan con las del modelo.
4. **Limites de recursos.** Define un tamano maximo de cuerpo y un limite de velocidad para evitar abusos y denegacion de servicio.
5. **CORS restringido.** Evita `allow_origins=["*"]` cuando la API es privada.
6. **Documentacion.** `/docs` y `/redoc` son comodas para ensenar; en produccion abierta al publico considera deshabilitarlas (`docs_url=None`, `redoc_url=None`) o protegerlas.
7. **Versionado y trazabilidad.** Registra `MODEL_NAME` y `MODEL_VERSION` en cada respuesta para auditar que version genero cada prediccion.
8. **Actualizaciones.** Antes de reemplazar `models/modelo.pkl`, valida el nuevo modelo en un entorno de pruebas.

---

## 18. Limitaciones conocidas

- **Un solo registro por solicitud.** `POST /predict` no procesa lotes (no hay `POST /predict/batch`).
- **Modelo en memoria del proceso.** Con varios *workers*, cada uno carga su propia copia del `.pkl` (consume mas memoria).
- **Sin autenticacion** ni control de cuotas.
- **Sin reentrenamiento en linea** ni monitoreo de *data drift*.
- **Esquema acoplado al modelo de ejemplo.** Los campos de `IrisFeatures` corresponden al modelo Iris; al cambiar de conjunto de datos hay que editar `schemas.py`.
- **Sin base de datos** ni persistencia de predicciones (solo logs).
- **Sin soporte para variables categoricas** en el ejemplo (el conjunto Iris es numerico). La [seccion 8.3](#83-columntransformer-codificadores-y-escaladores-por-que-reutilizar-y-no-reajustar) explica que hacer si las hay.
- **Sin pipeline de pruebas en CI** (se configuraria aparte, en GitHub Actions).

---

## 19. Lista de verificacion

Marca cada punto para confirmar que el proyecto funciona:

**Entorno**

- [ ] `python --version` muestra 3.11 o superior.
- [ ] El entorno virtual esta creado y activado (`.venv`).
- [ ] `pip install -e ".[dev,notebooks]"` termino sin errores.
- [ ] `python -c "import fastapi, sklearn, joblib, pandas, numpy"` no lanza errores.

**Modelo**

- [ ] `models/modelo.pkl` existe.
- [ ] `joblib.load("models/modelo.pkl")` funciona en el entorno virtual.
- [ ] El cuaderno 02 muestra `Accuracy` y `Las predicciones coinciden: True`.
- [ ] Anotaste las versiones de `scikit-learn`, `numpy` y `joblib` usadas al entrenar.

**API en local**

- [ ] `uvicorn app.main:app --reload` arranca sin errores.
- [ ] Los logs muestran el modelo cargado **una sola vez**.
- [ ] `GET /` responde `200` con el nombre del proyecto.
- [ ] `GET /health` responde `"model_loaded": true`.
- [ ] `POST /predict` con los valores de ejemplo responde `200` con `prediction` y `probabilities`.
- [ ] `POST /predict` sin `petal_width` responde `422`.
- [ ] `POST /predict` con `"sepal_length": "alto"` responde `422`.
- [ ] `POST /predict` con `"petal_width": -1` responde `422`.
- [ ] `/docs`, `/redoc` y `/openapi.json` responden `200`.
- [ ] Si renombras el `.pkl`, `/health` muestra `degraded` y `/predict` responde `503` sin filtrar rutas internas.

**Pruebas**

- [ ] `pytest` termina con `26 passed`.

**Despliegue (cuando corresponda)**

- [ ] El repositorio esta en GitHub y conectado al proveedor.
- [ ] El comando de inicio usa `${PORT}`.
- [ ] Las variables de entorno (`MODEL_PATH`, `MODEL_NAME`, `MODEL_VERSION`) estan configuradas.
- [ ] Los logs de produccion muestran el modelo cargado correctamente.
- [ ] `GET /health` en produccion devuelve `"model_loaded": true`.
- [ ] `POST /predict` en produccion devuelve una prediccion valida.

---

## Estado de verificacion de este repositorio

Todo lo siguiente fue **ejecutado y comprobado** al construir el proyecto:

| Verificacion | Resultado |
|---|---|
| Instalacion con `uv` (Python 3.11.14, scikit-learn 1.9.1) | Correcta |
| Ejecucion del cuaderno 01 (analisis descriptivo) | Correcta, con salidas y graficos |
| Ejecucion del cuaderno 02 (entrenamiento y guardado) | Correcta, `accuracy = 0.9333`, `.pkl` de 2.1 KB |
| `pytest` | `26 passed` |
| `GET /` | `200` |
| `GET /health` | `200`, `model_loaded: true`, 4 variables, 3 clases |
| `POST /predict` (setosa) | `200`, `setosa` con 0.9808 de probabilidad |
| `POST /predict` (virginica) | `200`, `virginica` con 0.9565 de probabilidad |
| `POST /predict` (campo ausente / tipo incorrecto) | `422` |
| `/docs`, `/redoc`, `/openapi.json` | `200` |

**No verificado:** el despliegue en un proveedor de hosting (se documenta el procedimiento, pero no se ejecuto en ningun servicio externo).
