"""anysearch2 —— 多源检索结果融合（Reciprocal Rank Fusion）。

把多个检索器（provider）各自返回的有序结果列表合并成一个统一排名，
只依赖标准库，便于嵌入到任何 Python 项目里。
"""

__version__ = "1.0.0"

from .fusion import FusionResult, reciprocal_rank_fusion
from .providers import Document, LocalCorpusProvider, Provider, SearchHit

__all__ = [
    "Document",
    "FusionResult",
    "LocalCorpusProvider",
    "Provider",
    "SearchHit",
    "reciprocal_rank_fusion",
    "__version__",
]
