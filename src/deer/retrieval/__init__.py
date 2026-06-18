"""Label-guided retrieval (DEER eq.1-5) + embedder interface."""
from .label_guided import LabelGuidedRetriever, MockEmbedder

__all__ = ["LabelGuidedRetriever", "MockEmbedder"]
