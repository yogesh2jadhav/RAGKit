"""
Purpose
-------
Represents failures while interacting with an
external service or infrastructure component.

Responsibilities
----------------
- Provide a common base class for service-related
  exceptions.

Does NOT
--------
- Represent validation or configuration errors.
"""

from ragkit.exceptions.ragkit_error import RagKitError


class ServiceError(RagKitError):
    """
    Base class for service-related errors.
    """