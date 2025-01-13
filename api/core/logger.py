import logging

from rich import print
from rich.logging import RichHandler

FORMAT = "%(message)s"
rich_handler = RichHandler(
    rich_tracebacks=True,
    show_time=True,
    show_level=True,
    show_path=True,
    markup=False,
)

logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[rich_handler]
)

# Configure logging for uvicorn access logs
# logging.getLogger("uvicorn.access").handlers = [rich_handler]
# logging.getLogger("uvicorn.error").handlers = [rich_handler]

log = logging.getLogger("rich")
