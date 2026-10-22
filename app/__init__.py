"""Paquete principal de la API del modelo de Machine Learning.

Organizado con arquitectura hexagonal (puertos y adaptadores):

- ``app.domain``: nucleo del negocio (entidades, puertos, errores). No depende
  de FastAPI, scikit-learn ni de ninguna otra tecnologia.
- ``app.application``: casos de uso que orquestan el dominio.
- ``app.adapters``: adaptadores de entrada (HTTP/FastAPI) y de salida
  (carga del ``.pkl`` con joblib/scikit-learn).
"""

__version__ = "0.1.0"
