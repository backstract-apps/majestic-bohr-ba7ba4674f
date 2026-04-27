"""
Worker Loader - Compiles and loads worker code in memory safely
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import builtins
import os
import sys
import importlib
import importlib.util

from loguru import logger


def _parse_iso_datetime_string(s: str) -> Optional[datetime]:
    """Parse ISO 8601-like string to datetime; returns None if not parseable."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    # Only attempt parse for strings that look like ISO date/datetime (avoid converting other fields)
    if len(s) < 10 or s[0].isdigit() is False or "-" not in s[:10]:
        return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _inject_worker_imports(globals_dict: Dict[str, Any]) -> None:
    """Inject common modules that generated worker code may use (storage, cache, http, env)."""
    globals_dict["os"] = os  # For os.getenv() - credentials from env (CACHE_*, QUEUE_*, etc.)
    try:
        import boto3
        globals_dict["boto3"] = boto3
    except ImportError:
        pass
    try:
        from botocore.exceptions import ClientError
        globals_dict["ClientError"] = ClientError
    except ImportError:
        pass
    try:
        import httpx
        globals_dict["httpx"] = httpx
    except ImportError:
        pass
    try:
        import redis
        globals_dict["redis"] = redis
    except ImportError:
        pass


def _ensure_datetime_in_payload(obj: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
    """
    Recursively convert ISO datetime strings in job payload to datetime objects.
    SQLite/SQLAlchemy DateTime columns require Python datetime/date, not strings.
    """
    if isinstance(obj, dict):
        return {k: _ensure_datetime_in_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ensure_datetime_in_payload(v) for v in obj]
    if isinstance(obj, str):
        parsed = _parse_iso_datetime_string(obj)
        if parsed is not None:
            return parsed
    return obj


class WorkerLoader:
    """Loads and compiles worker code with safe execution environment"""
    
    def __init__(self, app_root: Optional[str] = None):
        self.compiled_workers: Dict[str, Any] = {}
        # Safe builtins - essential functions, exception types (for try/except), and __import__
        self.SAFE_BUILTINS = {
            "len": len,
            "int": int,
            "str": str,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "print": print,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "__import__": builtins.__import__,
            # Exception types so generated worker code can use try/except Exception, etc.
            "BaseException": BaseException,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "AttributeError": AttributeError,
            "IndexError": IndexError,
            "RuntimeError": RuntimeError,
            "OSError": OSError,
        }

        # Preload generated app modules so workers can use models / DB / resources
        if app_root is None:
            try:
                app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            except Exception:
                app_root = None
        if app_root and app_root not in sys.path:
            sys.path.insert(0, app_root)

        # Default to None; worker code should handle missing dependencies gracefully
        self._models = None
        self._SessionLocal = None
        self._resource_manager = None

        # Ensure sqlite+libsql dialect is registered (required when app uses Turso/libsql)
        try:
            import sqlalchemy_libsql  # noqa: F401 - registers sqlite.libsql dialect
        except ImportError:
            pass

        # Load database first (so models/resources can import it if needed), then models.
        # When app_root is set (e.g. in child process), load by path so we always get the app's modules.
        if app_root and os.path.isfile(os.path.join(app_root, "database.py")):
            try:
                spec = importlib.util.spec_from_file_location("database", os.path.join(app_root, "database.py"))
                if spec and spec.loader:
                    database_module = importlib.util.module_from_spec(spec)
                    sys.modules["database"] = database_module
                    spec.loader.exec_module(database_module)
                    self._SessionLocal = getattr(database_module, "SessionLocal", None)
            except Exception as e:
                logger.warning(f"Could not load database from app_root {app_root}: {e}")
                self._SessionLocal = None
        if self._SessionLocal is None:
            try:
                database_module = importlib.import_module("database")
                self._SessionLocal = getattr(database_module, "SessionLocal", None)
            except Exception as e:
                logger.debug(f"Could not import database: {e}")
                self._SessionLocal = None

        if app_root and os.path.isfile(os.path.join(app_root, "models.py")):
            try:
                spec = importlib.util.spec_from_file_location("models", os.path.join(app_root, "models.py"))
                if spec and spec.loader:
                    models_module = importlib.util.module_from_spec(spec)
                    sys.modules["models"] = models_module
                    spec.loader.exec_module(models_module)
                    self._models = models_module
            except Exception as e:
                logger.warning(f"Could not load models from app_root {app_root}: {e}")
                self._models = None
        if self._models is None:
            try:
                self._models = importlib.import_module("models")
            except Exception as e:
                logger.debug(f"Could not import models: {e}")
                self._models = None

        if self._SessionLocal is None:
            logger.warning("SessionLocal not available; worker code using db/session will fail")

        # Prefer shared ResourceManager instance from routes if available
        try:
            from routes import resource_manager as _resource_manager  # type: ignore
            self._resource_manager = _resource_manager
        except Exception:
            try:
                # Fallback: create a new ResourceManager instance from resources.py
                resources_module = importlib.import_module("resources")
                ResourceManager = getattr(resources_module, "ResourceManager", None)
                self._resource_manager = ResourceManager() if ResourceManager else None
            except Exception:
                self._resource_manager = None
    
    def load_worker(self, worker_data: Dict[str, Any]):
        """Load and compile a worker with safe execution environment"""
        worker_id = worker_data["worker_id"]
        worker_code = worker_data["worker_code"]
        function_name = worker_data["function_name"]
        language = worker_data.get("language", "python").lower()
        
        if language == "python":
            # 1️⃣ Compile the code
            compiled_code = compile(worker_code, f"<worker_{worker_id}>", "exec")
            
            # 2️⃣ Load with safe builtins and injected app dependencies
            globals_dict: Dict[str, Any] = {
                "__builtins__": self.SAFE_BUILTINS,
            }

            # Inject generated app modules so worker code can use them directly
            if self._models is not None:
                globals_dict["models"] = self._models
            if self._SessionLocal is not None:
                globals_dict["SessionLocal"] = self._SessionLocal
            if self._resource_manager is not None:
                # Expose as both resource_manager and resources for convenience
                globals_dict["resource_manager"] = self._resource_manager
                globals_dict["resources"] = self._resource_manager

            # Inject common modules that generated worker code may use (storage, cache, http)
            _inject_worker_imports(globals_dict)

            exec(compiled_code, globals_dict)
            
            # 3️⃣ Extract the handler function
            if function_name not in globals_dict:
                raise ValueError(f"Function '{function_name}' not found in worker code")
            
            handler = globals_dict[function_name]
            
            self.compiled_workers[worker_id] = {
                "function": handler,
                "code": worker_code,
                "globals_dict": globals_dict,
            }
        else:
            # Store code for other languages
            self.compiled_workers[worker_id] = {
                "code": worker_code,
                "language": language,
            }
    
    def get_worker_function(self, worker_id: str):
        """Get compiled worker function"""
        if worker_id not in self.compiled_workers:
            raise ValueError(f"Worker {worker_id} not loaded")
        
        worker = self.compiled_workers[worker_id]
        if "function" in worker:
            return worker["function"]
        else:
            raise ValueError(f"Worker {worker_id} is not a Python worker")
    
    def execute_job(self, worker_id: str, job_data: Dict[str, Any]):
        """Execute a job with the worker function"""
        handler = self.get_worker_function(worker_id)
        # Convert ISO datetime strings in payload to datetime so SQLite/SQLAlchemy accept them
        job_data = _ensure_datetime_in_payload(job_data)
        return handler(job_data)