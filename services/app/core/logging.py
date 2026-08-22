from __future__ import annotations

import logging
import sys
from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from fastapi import Request, Response

LOGGER_NAME = "stkb"
REQUEST_ID_HEADER = "X-Request-ID"


def configure_logging(level: str) -> None:
    """配置独立的服务日志，避免依赖 uvicorn access log 是否开启。"""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


async def log_http_request(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    logger = get_logger("http")
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid4().hex[:12]
    started = perf_counter()
    logger.info(
        "request.started request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request.failed request_id=%s method=%s path=%s duration_ms=%d",
            request_id,
            request.method,
            request.url.path,
            round((perf_counter() - started) * 1000),
        )
        raise
    response.headers[REQUEST_ID_HEADER] = request_id
    logger.info(
        "request.completed request_id=%s method=%s path=%s status_code=%d duration_ms=%d",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        round((perf_counter() - started) * 1000),
    )
    return response
