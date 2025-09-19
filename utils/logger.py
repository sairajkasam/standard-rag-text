import logging

# ANSI colors
COLORS = {
    "DEBUG": "\033[36m",  # cyan
    "INFO": "\033[34m",  # dark blue
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[1;31m",  # bold red
    "RESET": "\033[0m",
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        record.msg = f"{color}{record.msg}{reset}"
        return super().format(record)


def get_logger(name="fastapi_app"):
    handler = logging.StreamHandler()
    formatter = ColorFormatter(
        "%(asctime)s [%(levelname)-5s] %(name)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
