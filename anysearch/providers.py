"""检索器抽象与一个可离线运行的关键词检索器。

真实项目里 provider 通常包着 Elasticsearch、向量库或第三方搜索 API；
这里给出统一的 :class:`Provider` 契约，并实现一个纯 Python 的关键词检索器，
让整条融合链路可以离线、确定性地测试。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence, runtime_checkable

#: 英文/数字按整词切分，CJK 按单字切分（避免 \w 把整段中文吞成一个 token）。
_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """极简分词：英文按单词、中文按单字切分，全部转小写。"""
    return [token.lower() for token in _WORD_RE.findall(text)]


@dataclass(frozen=True)
class Document:
    """语料库中的一篇文档。``doc_id`` 在融合阶段用于去重。"""

    doc_id: str
    title: str = ""
    text: str = ""
    url: str = ""
    tags: tuple[str, ...] = ()

    def searchable_text(self) -> str:
        return f"{self.title} {' '.join(self.tags)} {self.text}".strip()


@dataclass(frozen=True)
class SearchHit:
    """某个 provider 返回的单条命中。``score`` 越大越相关，需在 [0, +inf) 内。"""

    doc_id: str
    score: float
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Provider(Protocol):
    """检索器契约：给它一个查询，返回按相关度降序排列的命中列表。"""

    name: str

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        ...


class LocalCorpusProvider:
    """基于词频-逆文档频率（TF-IDF）的本地语料检索器。

    纯内存、无网络、结果稳定，适合做融合逻辑的默认数据源。
    """

    def __init__(self, name: str, documents: Iterable[Document]) -> None:
        self.name = name
        self._documents: list[Document] = list(documents)
        self._tokens: list[list[str]] = [tokenize(d.searchable_text()) for d in self._documents]
        self._df: dict[str, int] = {}
        for tokens in self._tokens:
            for token in set(tokens):
                self._df[token] = self._df.get(token, 0) + 1

    def __len__(self) -> int:
        return len(self._documents)

    @property
    def documents(self) -> Sequence[Document]:
        return tuple(self._documents)

    def _idf(self, token: str) -> float:
        total = len(self._documents)
        df = self._df.get(token, 0)
        # 平滑处理，避免语料库较小时除零或权重为负。
        return math.log((total + 1) / (df + 1)) + 1.0

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        query_tokens = tokenize(query)
        if not query_tokens or not self._documents:
            return []

        scored: list[SearchHit] = []
        for document, tokens in zip(self._documents, self._tokens):
            if not tokens:
                continue
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            length_norm = 1.0 / math.sqrt(len(tokens))
            score = 0.0
            for token in query_tokens:
                tf = counts.get(token, 0)
                if tf:
                    score += (1.0 + math.log(tf)) * self._idf(token) * length_norm
            if score > 0:
                scored.append(
                    SearchHit(
                        doc_id=document.doc_id,
                        score=round(score, 6),
                        title=document.title,
                        url=document.url,
                        snippet=self._snippet(document, query_tokens),
                        source=self.name,
                    )
                )

        scored.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return scored[:limit]

    @staticmethod
    def _snippet(document: Document, query_tokens: Sequence[str], width: int = 80) -> str:
        text = document.text or document.title
        lowered = text.lower()
        for token in query_tokens:
            index = lowered.find(token)
            if index >= 0:
                start = max(0, index - width // 3)
                return ("…" if start else "") + text[start : start + width].strip()
        return text[:width].strip()
