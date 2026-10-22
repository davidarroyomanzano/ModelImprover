# Datos

## Estructura

~~~text
projects/<producto>/<turbina>/<datos>/data/
├── raw/<Clase>/*.wav      # entrenamiento + validación
├── test/<Clase>/*.wav     # hold-out, nunca se usa para entrenar ni tunear
├── interim/               # generado
└── processed/             # generado
data/external/             # datos de terceros (comunes)
~~~

La etiqueta de cada muestra es el nombre de su carpeta. `scripts/cleanup_data_dirs.py` limpia `data/` global sin referencias.

## Convención de nombres

Los ficheros reales siguen el patrón:

~~~text
<Clase>.<Máquina>.<Origen>.<timestamp>.<ext...>.<id>.<pod>.<segmento>.t<inicio>s[.p<pico>s].wav
~~~

| Campo | Ejemplo | Significado |
|---|---|---|
| Clase | `Good` | Etiqueta |
| Máquina | `ES1Y_012`, `ES78_005`, `ES78_014`, `ES78_015`, `ES76_007`, `ES76_008` | Aerogenerador de origen |
| Origen | `Impacts_hub`, `Impacts_pitch`, `autorecord`, `onrecord` | Canal o modo de captura |
| Segmento | `background001`, `event002` | Recorte de fondo o de evento |
| `t<inicio>s` | `t2.5000s` | Instante del recorte en el audio original |
| `p<pico>s` | `p0.3456s` | Posición del pico (solo eventos) |

Ficheros sintéticos: `Impacts_hub_sintetic_NNN`, `Impacts_pitch_sintetic_NNN`, `Tendon_rupture.syntheticNNN.wav`.

## Separación train/test

- `<datos>/data/test/` (dentro de `projects/<producto>/<turbina>/`) es un hold-out fijo gestionado por `ds_project.data.test_set`.
- Varios recortes proceden del mismo audio original (mismo timestamp). Para evitar fugas, todos los recortes de un mismo original deben quedar en el mismo split; `tests/test_duplicate_detection.py` y `tests/test_test_set.py` lo verifican.
- Los sintéticos de `Tendon_rupture` se reparten: `synthetic066`–`085` en test, el resto en raw.

## Balance de clases

| Clase | Volumen aproximado |
|---|---|
| `Good`, `Stopped` | Cientos a miles de muestras |
| `Impacts_yaw` | ~100+ |
| `Tendon_rupture` | ~100 (casi todas sintéticas) |
| `Impacts_hub`, `Impacts_pitch` | Decenas (mezcla real/sintético) |
| `Hydraulic_charge_start` | Decenas |
| `Hydraulic_charge_stop` | ~20–30 |

Recomendaciones: ponderar clases o sobremuestrear en entrenamiento, y evaluar con métricas macro (F1 macro, matriz de confusión) en lugar de accuracy.
