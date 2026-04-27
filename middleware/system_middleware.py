"""
System Middleware Configuration
Generated system-level middleware that applies to the entire application
"""

from fastapi import FastAPI
import os


def setup_system_middleware(app: FastAPI) -> FastAPI:
    """
    Setup system-level middleware (applied once during app startup)
    This function configures all system middleware for the application
    """

    # CORS Middleware

    from middleware.cors_middleware import setup_cors_middleware

    app = setup_cors_middleware(app)

    return app
