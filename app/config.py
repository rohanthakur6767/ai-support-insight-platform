from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    db_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'tickets.db'}"
    chroma_dir: str = str(PROJECT_ROOT / "data" / "chroma")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    log_level: str = "INFO"

    # Category taxonomy used by zero-shot classifier
    categories: list[str] = [
        "Shipping & Delivery",
        "Returns & Refunds",
        "Payment & Billing",
        "Product Defect / Quality",
        "Order Status",
        "Account & Login",
        "Promotions & Discounts",
        "Cancellation",
        "Other",
    ]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    Path(PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    return settings
