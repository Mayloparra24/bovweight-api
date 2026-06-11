from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.predict import router as predict_router

app = FastAPI(
    title="BovWeight CR API",
    description="Microservicio de estimacion de peso bovino con YOLOv8-seg y Depth Pro",
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(predict_router)


@app.exception_handler(ValueError)
async def value_error_handler(_request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "NO_BOVINE_DETECTED",
                "message": str(exc),
            },
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": f"Error interno del servidor: {str(exc)}",
            },
        },
    )
