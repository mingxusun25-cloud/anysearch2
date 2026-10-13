"""anysearch2 的测试：检索器行为、RRF 融合语义、CLI 输出。"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anysearch import (  # noqa: E402
    Document,
    LocalCorpusProvider,
    SearchHit,
    reciprocal_rank_fusion,
)
from anysearch.cli import build_demo_providers, load_corpus, parse_weights  # noqa: E402
from anysearch.fusion import fuse_providers  # noqa: E402
from anysearch.providers import tokenize  # noqa: E402


def hit(doc_id: str, score: float = 1.0, source: str = "s") -> SearchHit:
    return SearchHit(doc_id=doc_id, score=score, title=doc_id, source=source)


class TokenizeTests(unittest.TestCase):
    def test_splits_words_and_cjk(self):
        self.assertEqual(tokenize("Hello, 混合检索!"), ["hello", "混", "合", "检", "索"])

    def test_empty_text(self):
        self.assertEqual(tokenize("   "), [])


class LocalCorpusProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = LocalCorpusProvider(
            "lexical",
            [
                Document(doc_id="a", title="向量检索", text="向量检索与近似最近邻"),
                Document(doc_id="b", title="评测", text="nDCG 与 MRR 评测指标"),
                Document(doc_id="c", title="无关文档", text="今天天气不错"),
            ],
        )

    def test_returns_relevant_docs_only(self):
        hits = self.provider.search("向量 检索")
        self.assertEqual(hits[0].doc_id, "a")
        self.assertTrue(all(h.score > 0 for h in hits))

    def test_unrelated_query_returns_empty(self):
        # 单字切分下，任何共享汉字都会命中；这里选一组语料里完全不出现的字。
        self.assertEqual(self.provider.search("棒球"), [])

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.provider.search(""), [])

    def test_limit_is_respected(self):
        self.assertLessEqual(len(self.provider.search("评测 指标 向量", limit=1)), 1)

    def test_source_and_snippet_are_filled(self):
        top = self.provider.search("向量")[0]
        self.assertEqual(top.source, "lexical")
        self.assertTrue(top.snippet)

    def test_results_are_deterministic(self):
        first = [(h.doc_id, h.score) for h in self.provider.search("向量 评测")]
        second = [(h.doc_id, h.score) for h in self.provider.search("向量 评测")]
        self.assertEqual(first, second)

    def test_len_and_documents_property(self):
        self.assertEqual(len(self.provider), 3)
        self.assertEqual(len(self.provider.documents), 3)


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_single_list_keeps_rank_order(self):
        fused = reciprocal_rank_fusion({"a": [hit("x"), hit("y"), hit("z")]})
        self.assertEqual([r.doc_id for r in fused], ["x", "y", "z"])
        self.assertGreater(fused[0].score, fused[1].score)

    def test_document_ranked_high_by_both_providers_wins(self):
        fused = reciprocal_rank_fusion(
            {
                "lexical": [hit("x", source="lexical"), hit("y", source="lexical")],
                "vector": [hit("y", source="vector"), hit("x", source="vector")],
            }
        )
        # x 在两个检索器里都是第 1 名，应当排在最前。
        self.assertEqual(fused[0].doc_id, "x")
        self.assertEqual(fused[0].sources, ["lexical", "vector"])

    def test_score_matches_rrf_formula(self):
        fused = reciprocal_rank_fusion({"a": [hit("x")], "b": [hit("x")]}, k=60)
        self.assertAlmostEqual(fused[0].score, 2 * (1 / 61), places=6)

    def test_weights_shift_ranking(self):
        lists = {
            "lexical": [hit("x", source="lexical"), hit("y", source="lexical")],
            "vector": [hit("y", source="vector"), hit("z", source="vector")],
        }
        heavy_vector = reciprocal_rank_fusion(lists, weights={"vector": 5.0})
        self.assertEqual(heavy_vector[0].doc_id, "y")
        weights_by_source = {c.source: c.weight for c in heavy_vector[0].contributions}
        self.assertEqual(weights_by_source["vector"], 5.0)
        self.assertEqual(weights_by_source["lexical"], 1.0)

    def test_deduplicate_false_keeps_both_entries(self):
        lists = {"a": [hit("x")], "b": [hit("x")]}
        self.assertEqual(len(reciprocal_rank_fusion(lists)), 1)
        self.assertEqual(len(reciprocal_rank_fusion(lists, deduplicate=False)), 2)

    def test_duplicate_doc_inside_one_provider_counts_once(self):
        fused = reciprocal_rank_fusion({"a": [hit("x", 5.0), hit("x", 1.0)]})
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].score, 1 / 61, places=6)

    def test_limit_and_empty_inputs(self):
        self.assertEqual(reciprocal_rank_fusion({}), [])
        self.assertEqual(reciprocal_rank_fusion({"a": []}), [])
        self.assertEqual(len(reciprocal_rank_fusion({"a": [hit("x"), hit("y")]}, limit=1)), 1)

    def test_sequence_input_uses_hit_source(self):
        fused = reciprocal_rank_fusion([[hit("x", source="p1")], [hit("x", source="p2")]])
        self.assertEqual(fused[0].sources, ["p1", "p2"])

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({"a": [hit("x")]}, k=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({"a": [hit("x")]}, limit=-1)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({"a": [hit("x")]}, weights={"a": -2})

    def test_to_dict_shape(self):
        payload = reciprocal_rank_fusion({"a": [hit("x", source="a")]})[0].to_dict()
        self.assertEqual(payload["doc_id"], "x")
        self.assertEqual(payload["sources"], ["a"])
        self.assertIn("contributions", payload)


class FuseProvidersTests(unittest.TestCase):
    def test_end_to_end_on_demo_corpus(self):
        results = fuse_providers(build_demo_providers(), "向量 检索", limit=3)
        self.assertTrue(results)
        self.assertEqual(results[0].doc_id, "hybrid-search")
        self.assertTrue({"fulltext", "title"} & set(results[0].sources))


class CliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "anysearch", *args],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=False,
        )

    def test_table_output(self):
        proc = self.run_cli("向量检索")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("score", proc.stdout)
        # 表格展示标题而非 doc_id，混合检索文档的标题即为此。
        self.assertIn("混合检索", proc.stdout)

    def test_json_output(self):
        proc = self.run_cli("rrf 融合", "--json", "--limit", "3")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["query"], "rrf 融合")
        self.assertLessEqual(len(payload["results"]), 3)
        self.assertIn("rrf-paper", [r["doc_id"] for r in payload["results"]])

    def test_no_hit_exits_one(self):
        proc = self.run_cli("zzzzz-棒球")
        self.assertEqual(proc.returncode, 1)

    def test_bad_weights_exits_two(self):
        proc = self.run_cli("rrf", "--weights", "broken")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("anysearch:", proc.stderr)

    def test_missing_corpus_exits_two(self):
        proc = self.run_cli("rrf", "--corpus", "no-such-file.json")
        self.assertEqual(proc.returncode, 2)

    def test_parse_weights(self):
        self.assertEqual(parse_weights("a=1,b=2.5"), {"a": 1.0, "b": 2.5})
        self.assertIsNone(parse_weights(None))

    def test_load_corpus_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            path.write_text(
                json.dumps({"documents": [{"doc_id": "d1", "title": "t", "tags": ["x"]}]}),
                encoding="utf-8",
            )
            docs = load_corpus(path)
            self.assertEqual(docs[0].doc_id, "d1")
            self.assertEqual(docs[0].tags, ("x",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
