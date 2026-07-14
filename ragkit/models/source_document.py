#__future__ lets you use features from newer Python versions before they become the default behavior.
# It provides easy way to  define class
from __future__ import annotations
from dataclasses import dataclass, field

from dataclasses import dataclass
from typing import Any
"""
 => SourceDocument is just like DTO in java which will be use in code.
"""
@dataclass(frozen=True, slots=True)
class SourceDocument:
    """
    Represents a document obtained from a source before parsing.
    """
    uri: str
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)