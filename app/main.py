import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.db.session import init_db, SessionLocal
from app.api.router import root_router
from app.api.v1.recognize import matcher

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI Server Startup: Initializing database & preloading RAM cache...")
    init_db()
    
    # Preload RAM Cache for Matcher Engine
    db = SessionLocal()
    try:
        count = matcher.load_cache(db)
        logger.info(f"Loaded {count} embeddings into RAM Cache.")
    except Exception as e:
        logger.error(f"Error loading RAM cache during startup: {e}")
    finally:
        db.close()
        
    yield
    logger.info("FastAPI Server Shutdown.")

app = FastAPI(
    title="AI Camera Core API",
    description="Core Backend & REST API for AI Camera Face Recognition System (Milestone Tuần 5)",
    version="1.0.0",
    openapi_version="3.0.3",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(root_router)

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        openapi_version="3.0.2"
    )
    try:
        schemas = openapi_schema.get("components", {}).get("schemas", {})
        for schema_name, schema_body in schemas.items():
            if "properties" in schema_body:
                for prop_name, prop_data in schema_body["properties"].items():
                    if prop_name in ["image", "images"]:
                        if prop_data.get("type") == "array":
                            prop_data["items"] = {"type": "string", "format": "binary"}
                        else:
                            prop_data["type"] = "string"
                            prop_data["format"] = "binary"
    except Exception as e:
        logger.error(f"Error customizing openapi schema: {e}")

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Serve Frontend Debug Dashboard statically
static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "online",
        "active_ai_combo": settings.ACTIVE_AI_COMBO,
        "cached_vectors_count": len(matcher.cache_profiles)
    }
