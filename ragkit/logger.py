"""
Purpose
-------
Provide the central logger used throughout RAGKit.

Responsibilities
----------------
- Create the framework logger.
- Provide a shared logger instance.

Does NOT
--------
- Configure logging.
- Write log files.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "ragkit"


logger = logging.getLogger(
    LOGGER_NAME,
)

#
# Prevent "No handler found" warnings
# if the application has not configured logging.
#
logger.addHandler(
    logging.NullHandler(),
)