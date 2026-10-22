# Guía de contribución

## Flujo

1. Crear una rama desde la principal.
2. Instalar hooks: `poetry run pre-commit install`.
3. Implementar cambios con tests.
4. Verificar:

~~~bash
poetry run python scripts/lint_all.py
poetry run pytest
~~~

## Reglas

- Estilo y lint con Ruff (ver `pyproject.toml` y `.pre-commit-config.yaml`).
- Sin rutas absolutas; usar rutas relativas a la raíz del proyecto.
- No versionar `.env`, cachés ni datos, modelos o exports de `projects/` (solo `project.yaml` y `templates/`).
- No añadir ficheros con nombres anómalos en la raíz (`tests/test_root_filenames.py`).
- No duplicar tests (`tests/test_no_duplicate_tests.py`).
- Nuevos datos: respetar la convención de nombres y no mezclar recortes del mismo original entre `raw` y `test`.
- Actualizar `docs/changelog.md` en cada cambio relevante.
