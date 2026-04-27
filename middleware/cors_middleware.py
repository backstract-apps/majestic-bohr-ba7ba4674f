"""
CORS Middleware Configuration
Generated CORS middleware for the application
"""

from fastapi import FastAPI
import os


def setup_cors_middleware(app: FastAPI) -> FastAPI:
    """
    Setup CORS middleware with configuration from environment variables
    """
    from fastapi.middleware.cors import CORSMiddleware

    # Get CORS configuration from environment variables
    origins = os.getenv("CORS_ORIGIN", "*").split(",")
    methods = os.getenv("CORS_METHOD", "GET,POST,PUT,DELETE,OPTIONS").split(",")
    allowed_headers = os.getenv("CORS_HEADERS", "*").split(",")
    exposed_headers = (
        os.getenv("CORS_EXPOSED_HEADERS", "").split(",")
        if os.getenv("CORS_EXPOSED_HEADERS")
        else []
    )
    credentials = os.getenv("CORS_CREDENTIALS", "false").lower() == "true"
    max_age = int(os.getenv("CORS_MAX_AGE", "3600"))

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=credentials,
        allow_methods=methods,
        allow_headers=allowed_headers,
        expose_headers=exposed_headers,
        max_age=max_age,
    )

    return app
