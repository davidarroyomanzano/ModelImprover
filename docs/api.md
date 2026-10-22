# API del paquete

Referencia de alto nivel de `ds_project`. Las firmas exactas están en los docstrings del código.

## Pipeline

~~~python
from ds_project import pipeline
# Ejecuta ingest → process → features → tuning → train → evaluate → export
# según configs/config.yaml (o configs/smoke.yaml en modo smoke).
~~~

## Datos

- `data.ingest`: lista ficheros WAV por clase y split.
- `data.process`: prepara el audio para el DSP.
- `data.test_set`: carga el hold-out y comprueba que no comparte originales con `raw`.

## DSP y features

- `dsp.params`: parámetros leídos de `configs/dsp.yaml`.
- `dsp.export`: escribe los parámetros DSP para el firmware.
- `features.audio`, `features.build`, `features.scaling`: extracción, ensamblado y escalado.

### Features disponibles

Se seleccionan por nombre en `features` de `configs/dsp.yaml`. Se calculan por frame y, en `mode: aggregate`, se resumen con `stats`.

| Nombre | Columnas | Descripción |
|---|---|---|
| `mfe` | `n_filters` | Bancos mel en dB cuantizados; no se combina con otras |
| `mfcc` | `n_mfcc` | MFCC (DCT sobre log-mel) |
| `log_mel` | `n_mels` | Espectrograma log-mel |
| `rms`, `zcr` | 1 | Energía RMS y tasa de cruces por cero |
| `spectral_centroid`, `spectral_bandwidth`, `spectral_flatness` | 1 | Descriptores espectrales |
| `spectral_rolloff` | 1 | Roll-off con `rolloff_percent` (0.85 por defecto) |
| `spectral_rolloff95` | 1 | Roll-off fijo al 95 % |
| `spectral_entropy`, `spectral_skewness`, `spectral_kurtosis` | 1 | Entropía, asimetría y curtosis del espectro de magnitud |
| `clipping` | 3 | `ratio`, `events`, `avg_length` de muestras con `abs(x) >= clipping_threshold` |
| `peak_to_rms`, `crest_factor` | 1 | Pico / RMS (ambas dan el mismo valor) |
| `energy_variance` | 1 | Varianza de la energía de 4 subventanas del frame |
| `harmonic` | 2 | `hnr` (dB) y `pitch` (Hz) por autocorrelación |
| `modulation` | 2 | `energy_delta` y `rms_delta` entre frames consecutivos |
| `cepstral` | `n_ceps` | log del espectro de potencia + DCT (3 por defecto) |
| `tkeo` | 2 | `mean` y `var` del operador de Teager-Kaiser |

Notas:
- El clipping se mide sobre la señal sin pre-emphasis; el resto usa la señal con pre-emphasis.
- `harmonic` limita la correlación normalizada a [0, 1) para evitar HNR infinito.
- Parámetros asociados: `n_ceps` y `clipping_threshold`.
- Las features añadidas en esta versión (todas salvo `mfe`, `mfcc`, `log_mel`, `rms`, `zcr` y las cuatro espectrales clásicas) solo están implementadas en Python, no en el firmware.

### Rama auxiliar (`aux_features`)

Con `features: [mfe]`, `aux_features` y `aux_stats` de `configs/dsp.yaml` definen un segundo vector de entrada: las features indicadas se calculan por frame y se agregan con `aux_stats`. `mfe` no puede usarse en `aux_features`. Si `aux_features` está vacío, el modelo tiene una sola entrada.

- `features.audio.extract_auxiliary_features` y `auxiliary_feature_names`: vector y nombres de la rama auxiliar.
- `features.build.build_auxiliary_features` y `build_auxiliary_features_cached`: ensamblado por lotes con caché.
- `DSPParams.aux_feature_dim` y `aux_feature_shape`: dimensión del vector auxiliar.

### Perfil `pinbolt`

`profile: pinbolt` (ver `configs/dsp_pinbolt.yaml`, se activa con `DS_DSP_CONFIG`) reproduce `templates/dsp_template.py`: `input_mfe` es el vector MFE aplanado (sin ventana ni escalado, cuantizado ×256) e `input_feats` son 25 features calculadas sobre ese vector y estandarizadas (`aux_features` debe ir vacío). Campos extra de `DSPParams`: `zero_padding`, `n_filters_final` (0 = filtros del primer bloque), y propiedades `mfe_flat_length` y `mfe_reshape`, con el que `build_model(..., mfe_reshape=...)` añade un `Reshape` antes de las Conv1D. Código en `ds_project.dsp.pinbolt`.

## Modelos

- `models.architectures`: construcción del modelo a partir de hiperparámetros. `build_model(..., aux_input_shape=...)` crea dos entradas: `input_mfe` (Conv1D, pooling y `Flatten`) e `input_feats` (Dense + Dropout), unidas con `Concatenate` antes de la clasificación.
- `models.tuning`: estudio Optuna configurado en `configs/optuna.yaml`.
- `models.train`, `models.evaluate`, `models.predict`: aceptan `X_train_feats`/`X_val_feats`/`X_feats` opcionales para la segunda entrada. `AudioClassifier` carga `feature_scaler.joblib` si el modelo usa `aux_features`. En TFLite las entradas se asocian por nombre (`input_mfe`, `input_feats`).
- `models.export`: modelo `.tflite` y cabecera C con el array del modelo. Para `int8`, `representative_data` puede ser una lista `[X_mfe, X_feats]`.

## Utilidades

- `utils.gpu`: detección y configuración de GPU.
- `utils.io`: lectura/escritura de audio y artefactos.
- `utils.storage`: gestión de directorios de salida.
- `visualization.plots`: curvas, matriz de confusión.
