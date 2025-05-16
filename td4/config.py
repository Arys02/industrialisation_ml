from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1] / 'td4'
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "td4"

LOGS_DIR = PROJ_ROOT / "logs"
LOGS_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'



