# Ejecución smoke

Ejecución mínima de extremo a extremo para validar el entorno y el pipeline en pocos minutos.

~~~bash
poetry run python scripts/smoke_check.py
~~~

Usa `configs/smoke.yaml`: subconjunto reducido de datos, pocos trials de Optuna y pocas épocas. Debe:

1. Cargar datos de `<datos>/data/raw` y `<datos>/data/test` del proyecto activo.
2. Extraer features y entrenar un modelo pequeño.
3. Exportar artefactos a `<datos>/models/`.

No sirve para medir rendimiento, solo para detectar roturas. Lo cubre `tests/test_smoke_check.py`.
