from pydantic_settings import BaseSettings, SettingsConfigDict
import os

# finds absolute env path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, ".env")


class Settings(BaseSettings):
    # security
    API_KEY: str = "your-super-secret-key"

    # app info
    FASTAPI_APP_NAME: str = "ReFinED Entity Linking API"
    FASTAPI_SERVER_PORT: int = 8002

    # mode config
    MODEL_DEVICE: str = "cpu"
    MOCK_MODE: bool = False

    # debug
    DEBUG: bool = False

    model_config = SettingsConfigDict(env_file=env_path, extra="ignore")

settings = Settings()

if settings.DEBUG:
    print(f"--- CONFIG LOADED ---")
    print(f"Device: {settings.MODEL_DEVICE}")
    print(f"Mock:   {settings.MOCK_MODE}")
    print(f"Key:    {settings.API_KEY[:5]}...")
    print(f"---------------------")