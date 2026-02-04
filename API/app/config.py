from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Change these names to match what main.py is calling
    FASTAPI_APP_NAME: str = "ReFinED Entity Linking API"
    DEBUG: bool = True
    MODEL_DEVICE: str = "gpu"
    FASTAPI_SERVER_PORT: int = 8002

settings = Settings()