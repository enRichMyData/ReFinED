# main.py
import logging
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.endpoints.refined_api import router as refined_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.FASTAPI_APP_NAME,
    debug=settings.DEBUG,
    description="An API for entity linking using the ReFinED model.",
    version="1.0.0",
    docs_url="/",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ReFinED router
app.include_router(refined_router)


# unique ID middleware (from crocodile !)
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "{\"event\":\"request\",\"request_id\":\"%s\",\"method\":\"%s\",\"path\":\"%s\",\"status_code\":%s}",
        request_id, request.method, request.url.path, response.status_code,
    )
    return response


# Category: Health Check
# ======================================
@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "api": "ReFinE"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.FASTAPI_SERVER_PORT)