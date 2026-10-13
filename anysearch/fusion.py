"""Reciprocal Rank Fusion（RRF）与相关的排名聚合工具。

RRF 的核心思想非常朴素：不比较不同检索器的分数（它们通常不可比），
只用"排名"投票——

    score(d) = Σ_p  w_p / (k + rank_p(d))

其中 ``rank`` 从 1 开始，``k`` 是平滑常数（论文推荐 60）。
被多个检索器同时排在前面的文档会获得更高的融合分，因此天然抑制了
单一检索器的"异常高分"噪声。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .providers import SearchHit

DEFAULT_K = 60


@dataclass(frozen=True)
class Contribution:
    """某个检索器对融合结果的一次投票。"""

    source: str
    rank: int
    weight: float
    value: float

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "rank": self.rank,
            "weight": self.weight,
            "value": round(self.value, 6),
        }


@dataclass
class FusionResult:
    """融合之后的一条结果。"""

    doc_id: str
    score: float
    hits: list[SearchHit] = field(default_factory=list)
    contributions: list[Contribution] = field(default_factory=list)
    k: int = DEFAULT_K

    @property
    def primary(self) -> SearchHit | None:
        """分数最高的原始命中，用于取标题/摘要等展示字段。"""
        return max(self.hits, key=lambda hit: hit.score) if self.hits else None

    @property
    def sources(self) -> list[str]:
        """贡献过票的检索器名（按在该检索器内的排名升序）。"""
        return [c.source for c in sorted(self.contributions, key=lambda c: c.rank)]

    @property
    def title(self) -> str:
        hit = self.primary
        return (hit.title if hit and hit.title else self.doc_id)

    @property
    def url(self) -> str:
        hit = self.primary
        return hit.url if hit else ""

    @property
    def snippet(self) -> str:
        hit = self.primary
        return hit.snippet if hit else ""

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "sources": self.sources,
            "contributions": [c.to_dict() for c in self.contributions],
        }


def _normalize_inputs(
    result_lists: Mapping[str, Sequence[SearchHit]] | Sequence[Sequence[SearchHit]],
) -> list[tuple[str, Sequence[SearchHit]]]:
    if isinstance(result_lists, Mapping):
        return [(str(name), list(hits)) for name, hits in result_lists.items()]
    normalized: list[tuple[str, Sequence[SearchHit]]] = []
    for index, hits in enumerate(result_lists):
        hits = list(hits)
        source = hits[0].source if hits and hits[0].source else f"provider{index}"
        normalized.append((source, hits))
    return normalized


def reciprocal_rank_fusion(
    result_lists: Mapping[str, Sequence[SearchHit]] | Sequence[Sequence[SearchHit]],
    *,
    k: int = DEFAULT_K,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
    deduplicate: bool = True,
) -> list[FusionResult]:
    """融合多个检索结果列表，返回按融合分降序排列的 :class:`FusionResult`。

    参数
    ----
    result_lists:
        要么是 ``{检索器名: 命中列表}`` 的映射，要么是若干命中列表组成的序列
        （此时用 ``hit.source`` 或 ``provider0``/``provider1``… 作为名字）。
    k:
        RRF 平滑常数，必须为正。越大越"平权"，越小越突出头部结果。
    weights:
        检索器权重，缺省为 1.0；未出现在映射里的检索器按 1.0 处理。
    limit:
        只返回前 N 条融合结果。
    deduplicate:
        为真（默认）时按 ``doc_id`` 合并同一文档在不同检索器中的命中；
        为假时保留每个检索器的条目独立计分。
    """
    if k <= 0:
        raise ValueError(f"k 必须为正整数，收到 {k!r}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit 不能为负，收到 {limit!r}")
    if weights:
        for name, weight in weights.items():
            if weight < 0:
                raise ValueError(f"权重不能为负: {name}={weight!r}")

    fused: "OrderedDict[str, FusionResult]" = OrderedDict()
    weights = dict(weights or {})

    for source, hits in _normalize_inputs(result_lists):
        weight = float(weights.get(source, 1.0))
        seen_in_source: set[str] = set()
        for position, hit in enumerate(hits, start=1):
            key = hit.doc_id if deduplicate else f"{source}::{hit.doc_id}"
            if deduplicate and hit.doc_id in seen_in_source:
                # 同一检索器里重复出现的文档只按最好名次计一次分。
                continue
            seen_in_source.add(hit.doc_id)

            value = weight / (k + position)
            entry = fused.get(key)
            if entry is None:
                entry = FusionResult(
                    doc_id=hit.doc_id,
                    score=0.0,
                    hits=[hit],
                    contributions=[],
                    k=k,
                )
                fused[key] = entry
            else:
                entry.hits.append(hit)
            entry.contributions.append(
                Contribution(source=source, rank=position, weight=weight, value=value)
            )
            entry.score += value

    ranked = sorted(
        fused.values(),
        key=lambda r: (-r.score, r.doc_id),
    )
    for entry in ranked:
        entry.score = round(entry.score, 6)

    if limit is not None:
        ranked = ranked[:limit]
    return ranked


def fuse_providers(
    providers: Iterable[object],
    query: str,
    *,
    per_provider_limit: int = 10,
    k: int = DEFAULT_K,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusionResult]:
    """便捷函数：依次调用各 provider 的 ``search``，再融合其结果。"""
    result_lists: "OrderedDict[str, list[SearchHit]]" = OrderedDict()
    for provider in providers:
        name = getattr(provider, "name", type(provider).__name__)
        hits = provider.search(query, limit=per_provider_limit)  # type: ignore[attr-defined]
        result_lists[name] = list(hits)
    return reciprocal_rank_fusion(result_lists, k=k, weights=weights, limit=limit)
