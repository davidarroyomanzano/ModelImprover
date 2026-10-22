"""Estructura de carpetas por proyecto y versión."""

from pathlib import Path
from typing import List, Union

from ds_project.config import load_settings

PathLike = Union[str, Path]

OVERLAY_HEADER = """# Overlay del proyecto {project}: se superpone a configs/*.yaml.
# Secciones admitidas: paths, data, train, export, storage, optuna, dsp.
"""


def init_project(
    product: str = "",
    turbine: str = "all",
    data_version: str = "",
    root: PathLike = ".",
    export_version: str = "",
) -> List[Path]:
    """Crea ``projects/<producto>/<turbina>`` con datos, modelos, export y templates.

    Los datos y modelos cuelgan de ``<versión de datos>``. También crea
    ``export/<export_version>`` (opcional) y ``project.yaml`` si no existe. Devuelve lo creado.
    """
    root = Path(root)
    settings = load_settings(
        product=product or None, turbine=turbine or None, data_version=data_version or None
    )
    paths = settings.resolved_paths()
    project_dir = root / settings.project_dir()
    dirs = [root / v for k, v in vars(paths).items() if k != "projects_dir"]
    dirs += [project_dir / "export", project_dir / "templates"]
    if export_version:
        dirs.append(project_dir / "export" / export_version)
    created: List[Path] = []
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True)
            created.append(d)
    overlay = project_dir / "project.yaml"
    if not overlay.exists():
        overlay.write_text(OVERLAY_HEADER.format(project=settings.project_name), encoding="utf-8")
        created.append(overlay)
    return created
