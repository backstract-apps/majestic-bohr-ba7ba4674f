"""
Worker Supervisor for coll_f696cc86af994ff3b7a3facbeda06a42
Manages and executes workers from message queues
"""
import asyncio
import json
import multiprocessing
import os
import resource
import sys
from typing import Any, Dict, List, Optional

# Load .env from app root for local dev only. In pod, credentials come from env (injected at deployment):
# QUEUE_<resource_id>_*, CACHE_<resource_id>_*, BACKSTRACT_AUTH_TOKEN, etc. - same pattern as api_state_parser / resources YAML.
try:
    from dotenv import load_dotenv
    _app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_app_root, ".env"))
except Exception:
    pass

from loguru import logger

from workers.worker_loader import WorkerLoader
from workers.adapters.base_adapter import QueueAdapter, create_queue_adapter

# Import ResourceManager for fetching user-provisioned queue credentials
try:
    from resources import ResourceManager
    _resource_manager = ResourceManager()
except ImportError:
    logger.warning("ResourceManager not available - user-provisioned queues will use env vars fallback")
    _resource_manager = None


def _run_worker_job_with_limits(
    worker_data: Dict[str, Any],
    job_data: Dict[str, Any],
    job_id: str,
) -> None:
    """
    Top-level function used as multiprocessing target to run a worker job
    with resource limits. Must be at module scope to be picklable on macOS/Linux.
    """
    try:
        # Child process: ensure app root is on sys.path and cwd so WorkerLoader can import database, models, resources
        _here = os.path.abspath(os.path.dirname(__file__))
        _app_root = os.path.dirname(_here)
        # Ensure we have the dir that contains database.py and workers/
        if not os.path.isfile(os.path.join(_app_root, "database.py")):
            _app_root = os.path.dirname(_app_root)
        if _app_root not in sys.path:
            sys.path.insert(0, _app_root)
        try:
            os.chdir(_app_root)
        except Exception:
            pass

        # Reload .env in child process for local dev. In pod, creds come from env (CACHE_*, QUEUE_*, BACKSTRACT_AUTH_TOKEN).
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(_app_root, ".env"))
        except Exception:
            pass

        # Set memory limit (may fail with "current limit exceeds maximum" on some systems; worker still runs)
        max_memory_mb = worker_data.get("max_memory_mb", 512)
        max_memory_bytes = max_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
        except Exception as e:
            logger.warning(f"Could not set memory limit for worker {worker_data.get('worker_id')}: {e}")

        # Recreate loader and load worker code in child process (pass app root so imports succeed)
        loader = WorkerLoader(app_root=_app_root)
        try:
            loader.load_worker(worker_data)
        except Exception as e:
            logger.error(f"Error loading worker {worker_data.get('worker_id')} in child process: {e}")
            return

        # ResourceManager is not initialized in child process (main app startup runs in parent). Initialize here so worker code can use resources.get_resource_by_alias(), etc.
        _rm = getattr(loader, "_resource_manager", None)
        if _rm is not None and hasattr(_rm, "initialize") and callable(getattr(_rm, "initialize")):
            # Worker process: platform-provisioned resources use BACKSTRACT_AUTH_TOKEN (from env in pod). User-provisioned cache/queue use env (CACHE_<id>_*, QUEUE_<id>_*).
            _token = getattr(_rm, "auth_token", None) or os.getenv("BACKSTRACT_AUTH_TOKEN")
            if not _token:
                logger.warning(
                    "BACKSTRACT_AUTH_TOKEN not set in worker process; platform proxy (storage/cache) may fail. "
                    "In pod, inject via env; locally use .env."
                )
            try:
                if asyncio.iscoroutinefunction(_rm.initialize):
                    asyncio.run(_rm.initialize())
            except Exception as init_e:
                logger.warning(f"ResourceManager.initialize() in worker process failed: {init_e}")

        # Execute job safely
        result = None
        try:
            result = loader.execute_job(worker_data["worker_id"], job_data)
            # Handle async results
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            logger.info(f"Worker {worker_data['worker_id']} completed job {job_id}")
            if result is not None:
                logger.info(f"Job {job_id} return value: {result}")
        except Exception as e:
            logger.error(f"Error in worker process for {worker_data.get('worker_id')}: {e}")
            raise
    except Exception as e:
        logger.error(f"Fatal error running worker job {job_id}: {e}")


class WorkerSupervisor:
    """Supervisor that manages workers and executes them in child processes"""
    
    def __init__(self):
        self.worker_loader = WorkerLoader()
        self.queue_adapters: Dict[str, QueueAdapter] = {}
        self.active_subscriptions: Dict[str, str] = {}
        self.running_processes: Dict[str, multiprocessing.Process] = {}
        self.is_running = False
        
    async def start(self):
        """
        Start the worker supervisor.
        
        This connects to RabbitMQ (using credentials from environment)
        and starts listening to queues. No HTTP server or port is used.
        """
        logger.info("Starting Worker Supervisor (connecting to RabbitMQ...)")
        self.is_running = True
        
        # Load all workers (compiles code and subscribes to queues)
        await self._load_workers()
        
        logger.info("Worker Supervisor started - listening to queues for jobs")
    
    async def stop(self):
        """Stop the worker supervisor"""
        logger.info("Stopping Worker Supervisor...")
        self.is_running = False
        
        # Unsubscribe from all queues (keys may be worker_id or worker_id:queue_name)
        worker_ids = set(k.split(":")[0] for k in self.active_subscriptions.keys())
        for worker_id in worker_ids:
            await self._unsubscribe_worker(worker_id)
        
        # Disconnect adapters
        for adapter in self.queue_adapters.values():
            try:
                await adapter.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting adapter: {e}")
        
        # Terminate processes
        for job_id, process in list(self.running_processes.items()):
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
        
        logger.info("Worker Supervisor stopped")
    
    async def _load_workers(self):
        """Load all workers"""
        workers = [{'function_name': 'handler',
  'language': 'python',
  'max_cpu_percent': 100,
  'max_execution_time_seconds': 300,
  'max_memory_mb': 512,
  'name': 'Test Worker',
  'queue_name': 'test_queue_coll_f696cc86af994ff3b7a3facbeda06a42',
  'queue_resource_id': None,
  'queue_type': 'rabbitmq',
  'queue_use_localhost_defaults': True,
  'queues': [{'queue_name': 'test_queue_coll_f696cc86af994ff3b7a3facbeda06a42',
              'queue_resource_id': None,
              'queue_type': 'rabbitmq',
              'queue_use_localhost_defaults': True}],
  'worker_code': 'def handler(job):\n'
                 "    '''\n"
                 '    Dummy test worker that prints hello world and the name from job data\n'
                 "    '''\n"
                 '    name = job.get("name", "Unknown")\n'
                 '    print(f"Hello World! Name from job data: {name}")\n'
                 '    return {"message": f"Hello World! Processed job for: {name}", "status": "success"}',
  'worker_function_name': 'handler',
  'worker_id': 'test_worker_coll_f696cc86af994ff3b7a3facbeda06a42'}]
        
        for worker_data in workers:
            await self._load_worker(worker_data)
    
    async def _load_worker(self, worker_data: Dict[str, Any]):
        """Load a single worker; subscribe to all its queues (supports many platform-provisioned queues per worker)."""
        try:
            worker_id = worker_data["worker_id"]
            queues = worker_data.get("queues") or []
            if not queues and worker_data.get("queue_name"):
                queues = [{"queue_name": worker_data["queue_name"], "queue_type": worker_data["queue_type"], "queue_resource_id": worker_data.get("queue_resource_id"), "queue_use_localhost_defaults": worker_data.get("queue_use_localhost_defaults", True)}]

            # Compile worker code
            self.worker_loader.load_worker(worker_data)

            # Connect once per queue_type (for platform-provisioned, all queues share same broker)
            if queues:
                await self._connect_to_queue(worker_data, queues[0])

            for queue_config in queues:
                await self._subscribe_worker(worker_data, queue_config)

            logger.info(f"Loaded worker: {worker_id} ({len(queues)} queue(s))")
        except Exception as e:
            logger.error(f"Error loading worker {worker_data.get('worker_id')}: {e}")
    
    def _get_adapter_key(self, queue_type: str, credentials: Dict[str, Any]) -> str:
        """
        Generate a unique key for the adapter based on connection details.
        Allows reusing adapters for queues sharing the same broker/vhost.
        """
        return f"{queue_type}:{credentials.get('host')}:{credentials.get('port')}:{credentials.get('vhost')}:{credentials.get('username')}"

    async def _connect_to_queue(self, worker_data: Dict[str, Any], queue_config: Optional[Dict[str, Any]] = None):
        """
        Connect to message queue (RabbitMQ).
        Credentials are injected via environment variables at deployment time.
        queue_config: optional per-queue config (for multiple queues per worker); uses worker_data if None.
        """
        q = queue_config or worker_data
        queue_type = (q.get("queue_type") or worker_data.get("queue_type", "rabbitmq")).lower()

        # Get credentials from environment (injected by platform)
        credentials = await self._get_queue_credentials(worker_data, queue_config)
        adapter_key = self._get_adapter_key(queue_type, credentials)

        if adapter_key in self.queue_adapters:
            return  # Already connected

        logger.info(f"Connecting to {queue_type} at {credentials.get('host', 'localhost')}:{credentials.get('port', 5672)}")

        adapter = create_queue_adapter(queue_type)
        await adapter.connect(credentials, (queue_config or worker_data).get("queue_config"))

        self.queue_adapters[adapter_key] = adapter
        logger.info(f"Connected to {queue_type} successfully (key: {adapter_key})")

    async def _get_queue_credentials(self, worker_data: Dict[str, Any], queue_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get queue credentials. Platform-provisioned queues use localhost defaults; user-provisioned use ResourceManager."""
        q = queue_config or worker_data
        # When queue is platform-provisioned, use localhost defaults (injected at code generation)
        if q.get("queue_use_localhost_defaults", worker_data.get("queue_use_localhost_defaults")):
            return {
                "host": "localhost",
                "port": 5672,
                "username": "guest",
                "password": "guest",
                "vhost": "/",
                "ssl": False,
            }
        # User-provisioned: fetch credentials from ResourceManager (same as job creation)
        queue_resource_id = q.get("queue_resource_id") or worker_data.get("queue_resource_id")
        if queue_resource_id and _resource_manager:
            try:
                logger.info(f"[WORKER_SUPERVISOR] Fetching queue credentials for resource_id='{queue_resource_id}'")
                
                # Initialize ResourceManager if not already done (worker supervisor is a separate process)
                if not _resource_manager._initialized:
                    await _resource_manager.initialize()
                
                mq_resource_info = _resource_manager.get_resource_by_alias(queue_resource_id)
                mq_credentials = mq_resource_info.get("credentials", {})
                
                if mq_credentials:
                    credentials = {
                        "host": mq_credentials.get("host", "localhost"),
                        "port": int(mq_credentials.get("port", 5672)),
                        "username": mq_credentials.get("username", "guest"),
                        "password": mq_credentials.get("password", "guest"),
                        "vhost": mq_credentials.get("vhost", "/"),
                        "ssl": mq_credentials.get("ssl", False),
                    }
                    logger.info(f"[WORKER_SUPERVISOR] Using credentials: host={credentials['host']}, port={credentials['port']}, username={credentials['username']}, vhost={credentials['vhost']}, ssl={credentials['ssl']}")
                    return credentials
                else:
                    logger.warning(f"[WORKER_SUPERVISOR] No credentials found for queue resource {queue_resource_id}, falling back to env vars")
            except Exception as e:
                logger.error(f"[WORKER_SUPERVISOR] Error fetching queue credentials from ResourceManager: {e}, falling back to env vars")
            
            # Fallback to environment variables if ResourceManager fails
            credentials = {
                "host": os.getenv(f"QUEUE_{queue_resource_id}_HOST") or os.getenv("RABBITMQ_HOST", "localhost"),
                "port": int(os.getenv(f"QUEUE_{queue_resource_id}_PORT") or os.getenv("RABBITMQ_PORT", "5672")),
                "username": os.getenv(f"QUEUE_{queue_resource_id}_USERNAME") or os.getenv("RABBITMQ_USERNAME", "guest"),
                "password": os.getenv(f"QUEUE_{queue_resource_id}_PASSWORD") or os.getenv("RABBITMQ_PASSWORD", "guest"),
                "vhost": os.getenv(f"QUEUE_{queue_resource_id}_VHOST") or os.getenv("RABBITMQ_VHOST", "/"),
                "ssl": os.getenv(f"QUEUE_{queue_resource_id}_SSL", "false").lower() == "true",
            }
        else:
            # No resource_id, use default env vars
            credentials = {
                "host": os.getenv("RABBITMQ_HOST", "localhost"),
                "port": int(os.getenv("RABBITMQ_PORT", "5672")),
                "username": os.getenv("RABBITMQ_USERNAME", "guest"),
                "password": os.getenv("RABBITMQ_PASSWORD", "guest"),
                "vhost": os.getenv("RABBITMQ_VHOST", "/"),
                "ssl": os.getenv("RABBITMQ_SSL", "false").lower() == "true",
            }
        return credentials
    
    def _subscription_key(self, worker_id: str, queue_name: str) -> str:
        """Key for active_subscriptions to support multiple queues per worker."""
        return f"{worker_id}:{queue_name}"

    async def _subscribe_worker(self, worker_data: Dict[str, Any], queue_config: Dict[str, Any]):
        """Subscribe worker to one queue (same handler for all queues of this worker)."""
        worker_id = worker_data["worker_id"]
        queue_name = queue_config["queue_name"]
        queue_type = (queue_config.get("queue_type") or "rabbitmq").lower()
        sub_key = self._subscription_key(worker_id, queue_name)

        if sub_key in self.active_subscriptions:
            return

        # Resolve adapter by credentials
        credentials = await self._get_queue_credentials(worker_data, queue_config)
        adapter_key = self._get_adapter_key(queue_type, credentials)
        
        adapter = self.queue_adapters.get(adapter_key)
        if not adapter:
            await self._connect_to_queue(worker_data, queue_config)
            adapter = self.queue_adapters[adapter_key]

        async def message_handler(message_data: Dict[str, Any]):
            await self._handle_job(worker_data, message_data)

        await adapter.subscribe(queue_name, message_handler, queue_config.get("queue_config") or worker_data.get("queue_config"))
        self.active_subscriptions[sub_key] = queue_name

        logger.info(f"Subscribed worker {worker_id} to queue {queue_name}")

    async def _unsubscribe_worker(self, worker_id: str):
        """Unsubscribe worker from all its queues."""
        to_remove = [k for k in self.active_subscriptions if k == worker_id or k.startswith(worker_id + ":")]
        for sub_key in to_remove:
            del self.active_subscriptions[sub_key]
    
    async def _handle_job(self, worker_data: Dict[str, Any], job_data: Dict[str, Any]):
        """Handle a job by running worker in child process"""
        try:
            worker_id = worker_data["worker_id"]
            job_id = f"{worker_id}_{int(asyncio.get_event_loop().time() * 1000)}"
            
            # Use job_data directly as payload (producer sends raw payload)
            job_payload = job_data
            
            # If payload is a string, try to parse it as JSON (in case it was double-encoded or sent as JSON string)
            if isinstance(job_payload, str):
                try:
                    import json
                    parsed = json.loads(job_payload)
                    if isinstance(parsed, (dict, list)):
                        job_payload = parsed
                except (ValueError, TypeError):
                    pass
            
            logger.info(f"Handling job {job_id} for worker {worker_id}")
            
            # Run in child process with limits
            process = await self._run_worker_in_process(worker_data, job_payload, job_id)
            
            if process:
                self.running_processes[job_id] = process
                asyncio.create_task(self._monitor_process(job_id, process))
        except Exception as e:
            logger.error(f"Error handling job: {e}")
    
    async def _run_worker_in_process(
        self, worker_data: Dict[str, Any], job_data: Dict[str, Any], job_id: str
    ) -> Optional[multiprocessing.Process]:
        """Run worker in child process with resource limits (macOS/Linux-safe)."""
        process = multiprocessing.Process(
            target=_run_worker_job_with_limits,
            args=(worker_data, job_data, job_id),
            name=f"worker-{worker_data['worker_id']}-{job_id}",
        )
        process.start()
        
        # Set timeout
        max_time = worker_data.get("max_execution_time_seconds", 300)
        asyncio.create_task(self._timeout_process(process, job_id, max_time))
        
        return process
    
    async def _timeout_process(self, process: multiprocessing.Process, job_id: str, timeout: int):
        """Terminate process if it exceeds timeout"""
        await asyncio.sleep(timeout)
        if process.is_alive():
            logger.warning(f"Process {job_id} exceeded timeout, terminating")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            if job_id in self.running_processes:
                del self.running_processes[job_id]
    
    async def _monitor_process(self, job_id: str, process: multiprocessing.Process):
        """Monitor process and cleanup when done"""
        try:
            while process.is_alive():
                await asyncio.sleep(1)
            if job_id in self.running_processes:
                del self.running_processes[job_id]
        except Exception as e:
            logger.error(f"Error monitoring process: {e}")


async def main():
    """
    Main entry point for worker supervisor.
    
    This is a background process that:
    1. Gets credentials from environment (injected at deployment)
    2. Connects to RabbitMQ
    3. Listens to queues for jobs
    4. Executes workers in child processes when jobs arrive
    
    No HTTP server or port is needed - it's just a queue listener.
    """
    supervisor = WorkerSupervisor()
    try:
        logger.info("Starting Worker Supervisor (background process, no HTTP server)")
        await supervisor.start()
        
        # Keep running and listening to queues
        logger.info("Worker Supervisor is listening to queues...")
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    finally:
        await supervisor.stop()


if __name__ == "__main__":
    # Run as background process - no port, just listens to queues
    asyncio.run(main())