"""Workarounds for MTEB task + Hugging Face `datasets` API drift (in-venv task code)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_BELEBELE_PATCHED = False


def apply_mteb_hf_patches() -> None:
    """Idempotent: patch task classes that break with current `datasets` / Hub layouts."""
    global _BELEBELE_PATCHED
    if _BELEBELE_PATCHED:
        return
    _patch_belebele_retrieval()
    _BELEBELE_PATCHED = True


def _patch_belebele_retrieval() -> None:
    """`mteb/belebele` requires a config name per language; stock task calls load_dataset without it."""
    from datasets import load_dataset as hf_load_dataset

    from mteb.tasks.retrieval.multilingual.belebele_retrieval import BelebeleRetrieval

    _EVAL_SPLIT = "test"

    def load_data_fixed(self, **kwargs) -> None:
        if self.data_loaded:
            return

        configs_needed: set[str] = set()
        for lang_pair in self.hf_subsets:
            languages = self.metadata.eval_langs[lang_pair]
            for L in languages:
                configs_needed.add(L.replace("-", "_"))

        path = self.metadata.dataset["path"]
        rev = self.metadata.dataset.get("revision")
        ds_by_cfg: dict = {}
        for cfg in sorted(configs_needed):
            ddict = hf_load_dataset(path, cfg, revision=rev)
            ds_by_cfg[cfg] = ddict[_EVAL_SPLIT]
        self.dataset = ds_by_cfg

        self.queries = {lang_pair: {_EVAL_SPLIT: {}} for lang_pair in self.hf_subsets}
        self.corpus = {lang_pair: {_EVAL_SPLIT: {}} for lang_pair in self.hf_subsets}
        self.relevant_docs = {
            lang_pair: {_EVAL_SPLIT: {}} for lang_pair in self.hf_subsets
        }

        for lang_pair in self.hf_subsets:
            languages = self.metadata.eval_langs[lang_pair]
            lang_corpus, lang_question = (
                languages[0].replace("-", "_"),
                languages[1].replace("-", "_"),
            )
            ds_corpus = self.dataset[lang_corpus]
            ds_question = self.dataset[lang_question]

            question_ids = {}
            for row in ds_question:
                question = row["question"]
                if question not in question_ids:
                    question_ids[question] = len(question_ids)

            link_to_context_id = {}
            context_idx = 0
            for row in ds_corpus:
                if row["link"] not in link_to_context_id:
                    context_id = f"C{context_idx}"
                    link_to_context_id[row["link"]] = context_id
                    self.corpus[lang_pair][_EVAL_SPLIT][context_id] = {
                        "title": "",
                        "text": row["flores_passage"],
                    }
                    context_idx = context_idx + 1

            for row in ds_question:
                query = row["question"]
                query_id = f"Q{question_ids[query]}"
                self.queries[lang_pair][_EVAL_SPLIT][query_id] = query

                context_link = row["link"]
                context_id = link_to_context_id[context_link]
                if query_id not in self.relevant_docs[lang_pair][_EVAL_SPLIT]:
                    self.relevant_docs[lang_pair][_EVAL_SPLIT][query_id] = {}
                self.relevant_docs[lang_pair][_EVAL_SPLIT][query_id][context_id] = 1

        self.data_loaded = True

    BelebeleRetrieval.load_data = load_data_fixed  # type: ignore[method-assign]
    logger.debug("Patched BelebeleRetrieval.load_data for per-config mteb/belebele loads")
