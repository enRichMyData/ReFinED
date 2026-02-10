from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from app.config import settings

import logging


logger = logging.getLogger("app.security")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    # no API key
    if not api_key:
        logger.warning("Security: Access attempt without API key")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key missing"
        )

    # successful key
    if api_key == settings.API_KEY:
        logger.info(f"Security: API Key verified successfully. (Key suffix: ...{api_key[-4:]})")
        return api_key

    # wrong key
    logger.error(f"Security: Invalid API key attempt. Provided {api_key[:4]}***")
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials"
    )