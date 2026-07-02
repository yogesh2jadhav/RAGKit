'''
 __future__ lets you use features from newer Python versions before they become the default behavior.
 It provides easy way to  define class
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

'''
 This Class called Document will hold a chunk of a file.
 frozen=True  - To make this Document object immutable
 slots=Ture - Python stores attributes in a fixed structure instead of a dictionary. This is a performance optimization.
'''
@dataclass(frozen=True, slots=True)
class Document:
    """
    Represents a logical document after it has been loaded.
    """
    id: UUID
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)