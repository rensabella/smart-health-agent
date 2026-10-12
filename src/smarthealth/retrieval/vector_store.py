"""Vector database boundary used by the verified local knowledge index.

QdrantClient is the existing concrete implementation. This protocol makes its
required operations explicit without claiming a remote/distributed adapter.
"""

from typing import Any, Protocol

from qdrant_client import models


class VectorDatabase(Protocol):
    def get_collection(self, collection_name: str, **kwargs: Any) -> models.CollectionInfo: ...

    def scroll(self, collection_name: str, **kwargs: Any) -> tuple[list[models.Record], Any]: ...

    def query_points(self, collection_name: str, **kwargs: Any) -> models.QueryResponse: ...

    def close(self) -> None: ...
