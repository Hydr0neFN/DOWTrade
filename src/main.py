import logging
import uvicorn
from src.journal.scheduler import start_scheduler

logging.basicConfig(level=logging.INFO)

def main():
    start_scheduler()
    uvicorn.run("src.dashboard.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
