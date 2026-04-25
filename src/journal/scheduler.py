from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import logging
from src.journal.daily import generate_daily_journal
from src.config import Settings
from src.db.repo import Database

log = logging.getLogger(__name__)
scheduler = BackgroundScheduler()

def run_daily_journal():
    settings = Settings()
    db = Database(settings.db_path)
    date_str = datetime.now().strftime("%Y-%m-%d")
    log.info(f"Running daily journal for {date_str}")
    try:
        generate_daily_journal(date_str, db)
    except Exception as e:
        log.error(f"Error generating journal: {e}")
    finally:
        db.close()

def start_scheduler():
    scheduler.add_job(
        run_daily_journal,
        'cron',
        day_of_week='mon-fri',
        hour=16,
        minute=30,
        timezone='America/New_York'
    )
    scheduler.start()
    log.info("Journal scheduler started")
