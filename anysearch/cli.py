"""anysearch2 命令行入口：对内置演示语料跑一遍多源检索 + 融合。

用法::

    python -m anysearch "向量检索"                 # 用内置演示语料
    python -m anysearch "rrf" --json --limit 5     # 结构化输出
    python -m anysearch "ranking" --corpus my.json --weights title=2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .fusion import DEFAULT_K, FusionResult, fuse_providers
from .providers import Document, LocalCorpusProvider, Provider

DEMO_DOCUMENTS: tuple[dict, ...] = (
    {
        "doc_id": "rrf-paper",
        "title": "Reciprocal Rank Fusion Outperforms Condorcet",
        "url": "https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf",
        "tags": ["rrf", "fusion", "ranking"],
        "text": (
            "Cormack 等人提出 Reciprocal Rank Fusion，用排名倒数之和融合多个"
            "检索系统的结果，在 TREC 评测上稳定优于 Condorcet 融合，且无需调参。"
        ),
    },
    {
        "doc_id": "bm25",
        "title": "BM25 与词频饱和",
        "url": "https://en.wikipedia.org/wiki/Okapi_BM25",
        "tags": ["bm25", "tf-idf", "lexical"],
        "text": "BM25 是经典的词法检索打分函数，通过饱和函数抑制高频词项的影响，是全文检索的默认基线。",
    },
    {
        "doc_id": "hybrid-search",
        "title": "混合检索：词法 + 向量",
        "url": "https://example.com/hybrid-search",
        "tags": ["hybrid", "vector", "ranking"],
        "text": (
            "生产环境常用混合检索：词法检索负责精确匹配与稀有词，向量检索负责语义泛化，"
            "两者结果再通过 RRF 或加权归一化融合。"
        ),
    },
    {
        "doc_id": "vector-db",
        "title": "向量数据库入门",
        "url": "https://example.com/vector-db",
        "tags": ["vector", "ann", "hnsw"],
        "text": "向量数据库使用 HNSW、IVF 等近似最近邻索引，在召回率与延迟之间做权衡。",
    },
    {
        "doc_id": "tokenization",
        "title": "中文分词对检索召回的影响",
        "url": "https://example.com/tokenization",
        "tags": ["tokenization", "chinese", "recall"],
        "text": "中文检索里分词策略直接决定倒排索引的粒度：字粒度召回高但噪声大，词粒度精确但容易漏召回。",
    },
    {
        "doc_id": "eval-metrics",
        "title": "检索评测指标：nDCG、MRR、Recall@k",
        "url": "https://example.com/eval-metrics",
        "tags": ["evaluation", "ndcg", "mrr"],
        "text": "融合效果需要离线评测：nDCG 关注排序质量，MRR 关注首个相关结果的位置，Recall@k 关注召回上限。",
    },
)


def build_demo_providers(documents: Sequence[Document] | None = None) -> list[Provider]:
    """构造两个视角不同的检索器：全文索引与仅标题/标签索引。"""
    docs = list(documents) if documents is not None else [Document(**raw) for raw in DEMO_DOCUMENTS]
    fulltext = LocalCorpusProvider("fulltext", docs)
    title_only_docs = [
        Document(doc_id=d.doc_id, title=d.title, url=d.url, tags=d.tags, text="") for d in docs
    ]
    title_only = LocalCorpusProvider("title", title_only_docs)
    return [fulltext, title_only]


def load_corpus(path: str | Path) -> list[Document]:
    """从 JSON 文件加载语料：``[{"doc_id": ..., "title": ...}, ...]``。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("documents", [])
    if not isinstance(raw, list):
        raise ValueError("语料文件应为文档数组，或含 documents 数组的对象")
    documents: list[Document] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条语料不是对象")
        if "doc_id" not in item:
            raise ValueError(f"第 {index} 条语料缺少 doc_id")
        documents.append(
            Document(
                doc_id=str(item["doc_id"]),
                title=str(item.get("title", "")),
                text=str(item.get("text", "")),
                url=str(item.get("url", "")),
                tags=tuple(item.get("tags", ()) or ()),
            )
        )
    return documents


def parse_weights(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    weights: dict[str, float] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"权重格式应为 name=value，收到 {chunk!r}")
        name, _, value = chunk.partition("=")
        weights[name.strip()] = float(value)
    return weights


def format_table(results: Sequence[FusionResult]) -> str:
    if not results:
        return "（没有命中）"
    width = max(len(r.title) for r in results)
    width = min(max(width, 12), 48)
    lines = [f"{'#':>3}  {'score':>8}  {'title':<{width}}  sources", "-" * (20 + width)]
    for index, result in enumerate(results, start=1):
        title = result.title if len(result.title) <= width else result.title[: width - 1] + "…"
        lines.append(
            f"{index:>3}  {result.score:>8.5f}  {title:<{width}}  {','.join(result.sources)}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anysearch",
        description="多源检索结果融合（RRF）演示与调试工具",
    )
    parser.add_argument("query", help="查询串")
    parser.add_argument("--corpus", help="自定义语料 JSON 文件；缺省使用内置演示语料")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"RRF 平滑常数，默认 {DEFAULT_K}")
    parser.add_argument(
        "--weights",
        help="检索器权重，如 fulltext=1,title=2（逗号分隔）",
    )
    parser.add_argument("--per-provider-limit", type=int, default=10, help="每个检索器返回多少条")
    parser.add_argument("--limit", type=int, default=None, help="最终输出多少条")
    parser.add_argument("--json", action="store_true", help="输出结构化 JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        weights = parse_weights(args.weights)
        documents = load_corpus(args.corpus) if args.corpus else None
        providers = build_demo_providers(documents)
        results = fuse_providers(
            providers,
            args.query,
            per_provider_limit=args.per_provider_limit,
            k=args.k,
            weights=weights,
            limit=args.limit,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"anysearch: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "k": args.k,
                    "weights": weights or {},
                    "results": [r.to_dict() for r in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_table(results))
    return 0 if results else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
