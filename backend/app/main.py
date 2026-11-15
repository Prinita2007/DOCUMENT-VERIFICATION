from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import ALLOWED_ORIGINS
from app.database import validate_database
from app.routes.api import router
from app.services.document_analysis import tesseract_status


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_database()
    yield


app = FastAPI(
    title="Student Document Verification System",
    description=(
        "Document intake and review assistance. Automated scores are indicators only; "
        "they do not establish that a document is genuine or fraudulent."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/", tags=["health"])
def home() -> dict:
    return {"message": "Student Document Verification System is running"}


@app.get("/health", tags=["health"])
def health() -> dict:
    validate_database()
    return {
        "status": "healthy",
        "database": "connected",
        "ocr": tesseract_status(),
    }


app.include_router(router)


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    import logging

    logging.getLogger("document_verification").exception("Unhandled application error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "An unexpected server error occurred"})
