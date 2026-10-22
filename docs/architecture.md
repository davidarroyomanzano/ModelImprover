# Arquitectura

## Módulos

| Módulo | Responsabilidad |
|---|---|
| `ds_project.config` | Carga y validación de YAML y `.env` |
| `ds_project.logger` | Configuración de logging desde `configs/logging.yaml` |
| `ds_project.data.ingest` | Descubrimiento de WAV y etiquetas por carpeta |
| `ds_project.data.process` | Normalización, remuestreo, recorte/padding a `data/interim`, `data/processed` |
| `ds_project.data.test_set` | Gestión del hold-out `data/test` y control de fugas |
| `ds_project.dsp.params` | Parámetros DSP tipados |
| `ds_project.dsp.export` | Exportación de parámetros DSP para firmware |
| `ds_project.features.audio` | Extracción de features de audio |
| `ds_project.features.build` | Construcción de matrices de features por split |
| `ds_project.features.scaling` | Ajuste y aplicación de escalado |
| `ds_project.models.architectures` | Definición de modelos |
| `ds_project.models.tuning` | Búsqueda Optuna |
| `ds_project.models.train` | Entrenamiento final |
| `ds_project.models.evaluate` | Métricas y matriz de confusión |
| `ds_project.models.predict` | Inferencia |
| `ds_project.models.export` | TFLite + cabecera C |
| `ds_project.utils` | GPU, E/S, almacenamiento |
| `ds_project.visualization.plots` | Figuras de cada run (matriz de confusión, Optuna) |
| `ds_project.pipeline` | Orquestación de extremo a extremo |

## Flujo

~~~mermaid
sequenceDiagram
    participant P as pipeline
    participant D as data
    participant F as features
    participant T as tuning
    participant M as models
    participant E as export
    P->>D: ingest + process
    P->>F: MFE + features auxiliares, con escalado independiente (fit solo en train)
    P->>T: Optuna (validación)
    T-->>P: mejores hiperparámetros
    P->>M: train + evaluate (test hold-out)
    P->>E: TFLite, .h, DSP params
~~~

## Diagrama de la pipeline y búsqueda de parámetros

~~~mermaid
flowchart TD
    subgraph CFG["Configuración"]
        DSPY["configs/dsp.yaml<br/>features: [mfe] + 2 mfe_blocks<br/>aux_features + aux_stats"]
        OPT["configs/optuna.yaml<br/>search_space (red)<br/>dsp_search_space (2 mfe_blocks)"]
    end

    subgraph TRIAL["Optuna · cada trial (models/tuning.py)"]
        S1["Muestrea parámetros DSP<br/>campo_1, campo_2<br/>(banda alta / banda baja)"]
        S2["Muestrea hiperparámetros de red<br/>lr, n_conv, filtros, units,<br/>n_dense_layers, dropout..."]
        RES["resolve_dsp_settings<br/>reconstruye 2 MFEBlock"]
        S1 --> RES
    end

    DSPY --> RES
    OPT --> S1
    OPT --> S2

    subgraph FE["Extracción de features"]
        AUDIO["Audio → ventanas"]
        MFE["MFE (2 bloques apilados<br/>en eje temporal)"]
        AUX["Features auxiliares agregadas<br/>(mean, std)"]
        SC1["FeatureScaler MFE<br/>(scaler.joblib)"]
        SC2["FeatureScaler aux<br/>(feature_scaler.joblib)"]
        AUDIO --> MFE --> SC1
        AUDIO --> AUX --> SC2
    end

    RES --> MFE
    RES --> AUX

    subgraph NET["Modelo Keras (architectures.build_model)"]
        IM["input_mfe"]
        CNN["Reshape + Conv1D + Pooling<br/>+ Flatten"]
        IF["input_feats"]
        DEN["Dense + Dropout"]
        CC["Concatenate (feature_concat)"]
        OUT["Dense + Softmax"]
        IM --> CNN --> CC
        IF --> DEN --> CC
        CC --> OUT
    end

    SC1 --> IM
    SC2 --> IF
    S2 --> NET

    OUT --> EVAL["Evaluación en validación<br/>→ métrica objetivo del trial"]
    EVAL -->|mejor trial| BEST["Parámetros óptimos"]
    BEST --> FINAL["Entrenamiento final<br/>[X_mfe, X_feats]"]
    FINAL --> EXP["Exportación<br/>model.keras / model.tflite (2 entradas)<br/>scalers + dsp_params.json / .h"]
~~~

Optuna busca los parámetros de los bloques MFE y los hiperparámetros de la red; las
`aux_features` son fijas (vienen de `dsp.yaml`) y no se buscan.

## Principios

- El escalado se ajusta solo con datos de entrenamiento.
- El test hold-out no participa en el tuning.
- Los parámetros DSP exportados deben coincidir exactamente con los usados en entrenamiento. Las features añadidas recientemente (ver `docs/api.md`) aún no están implementadas en el firmware.
- Con `dsp.aux_features`, el clasificador tiene dos entradas: `input_mfe` pasa por la CNN y
  `input_feats` contiene las features auxiliares agregadas. Ambas ramas se concatenan antes de
  las capas de clasificación. Los artefactos incluyen un scaler por entrada.
- Sin rutas absolutas en el código (`tests/test_no_absolute_paths.py`).
