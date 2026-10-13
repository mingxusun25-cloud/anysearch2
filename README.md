# anysearch2

多源检索结果融合库，实现了 **RRF（Reciprocal Rank Fusion，排名倒数融合）**，
零第三方依赖，可直接嵌入现有检索服务。

不同检索器（词法、向量、第三方 API）返回的分数互不可比，直接加权求和经常翻车。
RRF 只使用**排名**投票：

```
score(d) = Σ_p  w_p / (k + rank_p(d))        # rank 从 1 开始，k 默认 60
```

被多个检索器同时排在前面的文档自然得分更高，从而抑制单个检索器的异常高分。

## 安装与运行

无需安装依赖，`python -m anysearch` 即可体验（内置演示语料）：

```console
$ python -m anysearch "向量检索"
  #     score  title                                         sources
----------------------------------------------------------------
  1   0.03279  混合检索：词法 + 向量                                  fulltext,title
  2   0.03226  向量数据库入门                                       fulltext,title
  3   0.03150  中文分词对检索召回的影响                                  fulltext,title
  4   0.03126  检索评测指标：nDCG、MRR、Recall@k                      title,fulltext
  5   0.01562  BM25 与词频饱和                                    fulltext
  6   0.01515  Reciprocal Rank Fusion Outperforms Condorcet  fulltext
```

调高标题检索器的权重，评测类文档被顶上来了：

```console
$ python -m anysearch "评测 指标" --weights fulltext=1,title=3 --limit 3
  #     score  title                                         sources
----------------------------------------------------------------
  1   0.06557  检索评测指标：nDCG、MRR、Recall@k                      fulltext,title
  2   0.01613  Reciprocal Rank Fusion Outperforms Condorcet  fulltext
```

结构化输出：

```console
$ python -m anysearch "rrf 融合" --json --limit 1
{
  "query": "rrf 融合",
  "k": 60,
  "weights": {},
  "results": [
    {
      "doc_id": "rrf-paper",
      "score": 0.032787,
      "title": "Reciprocal Rank Fusion Outperforms Condorcet",
      "url": "https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf",
      "snippet": "…procal Rank Fusion，用排名倒数之和融合多个检索系统的结果，在 TREC 评测上稳定优于 Condorcet 融合，且无需调参。",
      "sources": ["fulltext", "title"],
      "contributions": [
        { "source": "fulltext", "rank": 1, "weight": 1.0, "value": 0.016393 },
        { "source": "title", "rank": 1, "weight": 1.0, "value": 0.016393 }
      ]
    }
  ]
}
```

## 作为库使用

```python
from anysearch import Document, LocalCorpusProvider, reciprocal_rank_fusion

lexical = LocalCorpusProvider("lexical", [
    Document(doc_id="d1", title="向量检索", text="HNSW 近似最近邻"),
])
vector = LocalCorpusProvider("vector", [...])          # 也可换成你自己实现的 Provider

results = reciprocal_rank_fusion(
    {
        "lexical": lexical.search("向量检索", limit=10),
        "vector": vector.search("向量检索", limit=10),
    },
    k=60,
    weights={"lexical": 1.0, "vector": 1.5},
    limit=5,
)

for rank, item in enumerate(results, 1):
    print(rank, item.doc_id, round(item.score, 5), item.sources, item.title)
```

### 自有检索器接入

只要满足 `Provider` 协议（有 `name`，并实现 `search(query, *, limit) -> list[SearchHit]`），
就能直接交给 `fuse_providers()`：

```python
class ElasticProvider:
    name = "elastic"

    def search(self, query, *, limit=10):
        raw = es.search(index="docs", query=query, size=limit)
        return [
            SearchHit(doc_id=h["_id"], score=h["_score"], title=h["_source"]["title"])
            for h in raw["hits"]["hits"]
        ]
```

## 命令行参数

```
python -m anysearch query [--corpus FILE] [--k N] [--weights a=1,b=2]
                          [--per-provider-limit N] [--limit N] [--json]
```

`--corpus` 接收 `{"documents": [{"doc_id", "title", "text", "url", "tags"}]}`
或等价的文档数组，见 [`data/demo_corpus.json`](data/demo_corpus.json)。
退出码：`0` 有结果、`1` 无结果、`2` 参数或文件错误。

## 测试

```bash
python -m unittest discover -s tests -t .
```

27 个用例，覆盖分词、TF-IDF 检索器、RRF 公式与权重、去重语义、
序列/映射两种输入形式、参数校验与 CLI 输出。

## 实现要点

- **去重**：默认按 `doc_id` 合并同一文档的多次命中，同一检索器内重复出现只按最好名次计一次分
  （`deduplicate=False` 可关闭，用于调试"谁贡献了什么名次"）。
- **不比较分数**：只吃排名，因此不必做 min-max 归一化，也不用担心跨检索器的分数量纲。
- **可解释性**：每条结果都带 `contributions`，记录哪个检索器在第几名投了多少分，便于线上排查。
- **确定性**：同分时按 `doc_id` 兜底排序，保证结果可复现（测试依赖这一点）。

## 许可

MIT，见 [LICENSE](LICENSE)。
