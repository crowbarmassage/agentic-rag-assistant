"""Run script for ShopUNow Assistant API."""

from dotenv import load_dotenv
load_dotenv()  # Load .env before any other imports

import uvicorn
from src.config import get_settings


def main():
    settings = get_settings()

    print(f"Starting ShopUNow Assistant API")
    print(f"  Host: {settings.api_host}")
    print(f"  Port: {settings.api_port}")
    print(f"  Debug: {settings.debug}")
    print(f"  Docs: http://{settings.api_host}:{settings.api_port}/docs")

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )


if __name__ == "__main__":
    main()
