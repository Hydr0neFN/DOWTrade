import logging
import asyncio
import threading
import signal
import sys
import uvicorn
from src.journal.scheduler import start_scheduler
from src.live.runner import LiveRunner

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

runner = LiveRunner()
loop = asyncio.new_event_loop()

def run_live():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner.start())

def shutdown_handler(sig, frame):
    log.info("SIGTERM received, shutting down...")
    if loop.is_running():
        asyncio.run_coroutine_threadsafe(runner.stop(), loop)
    sys.exit(0)

def main():
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    start_scheduler()
    
    t = threading.Thread(target=run_live, daemon=True)
    t.start()
    
    uvicorn.run("src.dashboard.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
