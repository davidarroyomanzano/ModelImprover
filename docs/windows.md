# Notas para Windows

- Ejecutar los comandos con `poetry run` para usar el intérprete del entorno local (`.venv\Scripts\python.exe`). `tests/test_tasks_interpreter.py` verifica que `tasks.py` no dependa de un intérprete global.
- `.gitattributes` fija los finales de línea; no cambiar `core.autocrlf` en el repositorio.
- Algunos nombres de fichero son muy largos (p. ej. `data/raw/Good/...`). Activar rutas largas:

~~~powershell
git config --global core.longpaths true
~~~

  y habilitar `LongPathsEnabled` en el registro si es necesario.
- Hay nombres con espacios y paréntesis (`... 1786751284 (1).wav`); citar rutas en scripts de shell.
- El paralelismo de Optuna con procesos puede requerir el guard `if __name__ == "__main__":`.
