# BovWeight CR — Microservicio de IA

Microservicio de estimación de peso bovino usando YOLOv8-seg + Depth Pro.
Diseñado para ejecutarse en Google Colab y consumirse desde la API REST en Laravel.

## Endpoints

- `GET /health` — estado del servicio.
- `POST /api/v1/predict-weight` — recibe una imagen y una constante de raza, retorna el peso estimado.

## Ejecutar localmente

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Ejecutar en Colab

Abrir `Lanzador_Colab.ipynb` y seguir las celdas. Usa `pyngrok` + `nest-asyncio` para exponer el servidor.

## Patrones de diseño aplicados

| Patrón | Archivo(s) | Rol |
| ------ | ---------- | --- |
| **Adapter** | `app/infrastructure/yolo_segmenter.py`<br>`app/infrastructure/depth_pro_estimator.py` | Adaptan Ultralytics YOLO y Depth Pro a los protocolos del dominio. |
| **Strategy** | `app/domain/protocols.py`<br>`app/application/strategies/yolo_depth_pro_strategy.py`<br>`app/application/weight_estimator.py` | Permite cambiar el algoritmo de estimación sin modificar la cadena de validación. |
| **Factory Method** | `app/infrastructure/handlers/factory.py` | Crea y ensambla la cadena de validación. |
| **Chain of Responsibility** | `app/infrastructure/handlers/` | Valida la imagen paso a paso: formato → resolución → detección de bovino → estimación. |

## Tests

```bash
python -m unittest discover tests
```

## Estructura

```
app/
├── api/              # Routers FastAPI
├── application/      # Casos de uso y estrategias
├── domain/           # Protocolos y esquemas
├── infrastructure/   # Adaptadores de modelos y handlers de validación
└── main.py           # Punto de entrada
```
