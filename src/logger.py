import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    filename="example.log",
    encoding="utf-8",
    format="%(levelname)s:%(message)s",
)

# Create logger
logger = logging.getLogger(__name__)
