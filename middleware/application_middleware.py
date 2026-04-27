"""
Auto-generated middleware file
Contains middleware functions and groups defined in the collection
"""

import os
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, List, Dict, Any
import jwt
from clerk_backend_api import Clerk
from clerk_backend_api.security.types import AuthenticateRequestOptions
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from textwrap import dedent


# Middleware: CORS Middleware
# Slug: cors_middleware
async def cors_middleware(request: Request) -> Dict[str, Any]:
    """
    CORS Middleware
    Generated from middleware ID: mid_892813df3acb45d3a52c2d5521541f67
    """
    try:

        def setup_cors_middleware(app: FastAPI):
            """
            Setup CORS middleware with configuration from environment variables
            """

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

        return {}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Middleware error: {str(e)}")


# Middleware Group Dependency Functions


async def default_dependency(request: Request) -> Dict[str, Any]:
    """
    Dependency function for middleware group: default
    Executes all middlewares in the group in sequence
    """
    result = {}

    # Execute cors_middleware
    middleware_result = await cors_middleware(request)
    if isinstance(middleware_result, dict):
        result.update(middleware_result)
        # Store middleware variables in request.state for API handlers to access
        for key, value in middleware_result.items():
            setattr(request.state, key, value)

    return result
