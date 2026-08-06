import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes import router
from app.predictor import predictor

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mlops_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup hook: load ML models
    logger.info("Initializing FastAPI service & loading machine learning models...")
    predictor.load_models()
    yield
    # Shutdown hook
    logger.info("Shutting down FastAPI service...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Production-ready MLOps microservice providing real-time predictions "
        "for both Regression (California Housing) and Classification (Breast Cancer) models."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include endpoint routes
app.include_router(router, prefix=settings.API_PREFIX)


@app.get("/", tags=["General"])
def read_root():
    return {
        "message": "Welcome to the Dual-Model MLOps Microservice API",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "regression": "/predict/regression",
            "classification": "/predict/classification"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please check server logs."}
    )
