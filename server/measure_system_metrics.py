import logging
import os
import sys

# Add the server directory to Python path for imports
sys.path.insert(0, os.path.dirname(__file__))

from utils.measure_utils.cli import main

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        sys.exit(1)
