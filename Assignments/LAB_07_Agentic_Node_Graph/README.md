# Lab 7 - Agentic Node Graph

**Skill:** Swarms

**Worklet mapping:** Graph orchestration with conditional edges and checkpoints, two cooperating graphs rather than a linear script.

## What this demonstrates

Writing is a 9 node LangGraph with three conditional edges and SQLite checkpoints. The Dev Agent is a second, separate graph: plan, write a failing test, patch, test, at most 3 rounds, working in a throwaway git worktree.

## Real code exercised

The actual source files this lab is built on, copied in full below, not linked only.

### `services/research-writing/writing/graph.py`

```python
"""LangGraph orchestration and the package's public run interface."""

import asyncio
import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Overwrite, Send
from pydantic import BaseModel

from writing.adapters.artifact_store import (
    ArtifactStore,
    JsonValue,
    LocalArtifactStore,
)
from writing.adapters.document_parser import DocumentParser, contains_pdf
from writing.adapters.language_model import (
    LanguageModel,
    close_language_model,
    resolve_language_model,
)
from writing.config import ReviewConfig
from writing.modules.card_validation import (
    audit_literature_card_semantics,
    validate_literature_card,
)
from writing.modules.citation_formatter import build_citation_map
from writing.modules.document_registry import discover_documents
from writing.modules.hierarchical_recall import (
    analyze_tag_coverage,
    direct_items,
)
from writing.modules.literature_card import (
    build_cached_literature_card,
    build_literature_card,
)
from writing.modules.outline_parser import parse_outline
from writing.modules.outline_source import resolve_outline_source
from writing.modules.review_assembler import assemble_review
from writing.modules.review_auditor import (
    audit_full_review,
    audit_full_review_semantics,
    audit_visual_recommendation_evidence,
)
from writing.modules.review_writer import (
    revise_full_review,
    write_conclusion,
    write_introduction,
)
from writing.modules.section_auditor import (
    audit_section,
    audit_section_semantics,
)
from writing.modules.section_context import build_section_context
from writing.modules.section_writer import write_section_with_revisions
from writing.modules.tag_index import build_tag_index
from writing.modules.tag_semantics import define_tag_semantics
from writing.modules.tag_tree import build_tag_tree
from writing.modules.writing_order import plan_writing_order
from word_budget import allocate_section_budgets
from writing.schemas import (
    CardAudit,
    CitationInfo,
    FullReviewAudit,
    LiteratureCard,
    OutlineNode,
    PaperDocument,
    ReviewDocument,
    ReviewInput,
    ReviewResult,
    SectionAudit,
    SectionDraft,
    SectionRunResult,
    SectionSummary,
    SectionWritingContext,
    TagDefinition,
    TagIndex,
    TagIndexItem,
)
from writing.state import ReviewState

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _dump(value: BaseModel) -> JsonValue:
    return cast(JsonValue, value.model_dump(mode="json"))


def _dump_many(values: Sequence[BaseModel]) -> JsonValue:
    return cast(JsonValue, [value.model_dump(mode="json") for value in values])


def _load_many(
    store: ArtifactStore, path: str, model: type[BaseModel]
) -> list[BaseModel]:
    raw = store.read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"artifact must contain a list: {path}")
    return [model.model_validate(item) for item in raw]


def _load_tree(store: ArtifactStore, path: str) -> list[OutlineNode]:
    return cast(list[OutlineNode], _load_many(store, path, OutlineNode))


def _load_definitions(store: ArtifactStore, path: str) -> list[TagDefinition]:
    return cast(list[TagDefinition], _load_many(store, path, TagDefinition))


def _load_index(store: ArtifactStore, path: str) -> TagIndex:
    raw = store.read_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"artifact must contain an object: {path}")
    return {
        str(tag_id): [TagIndexItem.model_validate(item) for item in items]
        for tag_id, items in raw.items()
        if isinstance(items, list)
    }


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _reset_reduced_list(values: Sequence[str] = ()) -> list[str]:
    """Clear a reducer-backed channel at the start of a new invocation."""

    return cast(list[str], Overwrite(list(values)))


def _unresolved_section_draft(tag_id: str, output_language: str) -> SectionDraft:
    """Create safe body prose while the audit artifact retains the failed draft."""

    content = (
        "*本节审计未能解决，因此未纳入未经支持的内容。*"
        if output_language.casefold().startswith("zh")
        else "*This section audit could not be resolved; unsupported content was omitted.*"
    )
    return SectionDraft(
        tag_id=tag_id,
        content=content,
        cited_paper_ids=[],
        used_point_ids=[],
    )


def build_review_graph(
    model: LanguageModel,
    store: ArtifactStore,
    runtime: ReviewConfig,
    checkpointer: BaseCheckpointSaver[str],
) -> CompiledStateGraph[ReviewState, None, ReviewState, ReviewState]:
    """Compile the review graph around supplied model and persistence adapters."""

    async def initialize(state: ReviewState) -> dict[str, object]:
        tree = build_tag_tree(parse_outline(state["outline_source"]))
        tree_path = store.write_json("tag_tree.json", _dump_many(tree))
        return {"input_path": "input.json", "tag_tree_path": tree_path}

    async def define_tags(state: ReviewState) -> dict[str, object]:
        tree = _load_tree(store, state["tag_tree_path"])
        definitions = await define_tag_semantics(
            state["review_question"], tree, model
        )
        path = store.write_json("tag_definitions.json", _dump_many(definitions))
        return {"tag_definitions_path": path}

    async def register_documents(state: ReviewState) -> dict[str, object]:
        documents = discover_documents(state["literature_directory"])
        manifest = {
            "documents": [document.model_dump(mode="json") for document in documents]
        }
        path = store.write_json("manifest.json", cast(JsonValue, manifest))
        warnings = ["No Markdown literature documents were found."] if not documents else []
        return {
            "manifest_path": path,
            "paper_ids": [document.paper_id for document in documents],
            "warnings": warnings,
        }

    def dispatch_cards(state: ReviewState) -> str | list[Send]:
        raw = store.read_json(state["manifest_path"])
        if not isinstance(raw, dict):
            raise ValueError("manifest does not contain a document list")
        documents_value = raw.get("documents")
        if not isinstance(documents_value, list):
            raise ValueError("manifest does not contain a document list")
        documents: list[dict[str, JsonValue]] = []
        for document in documents_value:
            if not isinstance(document, dict):
                raise ValueError("manifest document entries must be objects")
            documents.append(document)
        if not documents:
            return "build_tag_index"
        return [
            Send(
                "build_literature_card",
                {
                    "worker_document": document,
                    "review_question": state["review_question"],
                    "tag_tree_path": state["tag_tree_path"],
                    "tag_definitions_path": state["tag_definitions_path"],
                },
            )
            for document in documents
        ]

    async def build_card_worker(state: ReviewState) -> dict[str, object]:
        document = PaperDocument.model_validate(state["worker_document"])
        audit_path = f"card_audits/{document.paper_id}.json"
        definitions = _load_definitions(store, state["tag_definitions_path"])
        tree = _load_tree(store, state["tag_tree_path"])
        attempt_errors: list[str] = []
        for attempt in range(1, runtime.max_card_attempts + 1):
            try:
                if attempt == 1:
                    card = await build_cached_literature_card(
                        state["review_question"], definitions, document, model, store
                    )
                else:
                    # Targeted retry (DECISIONS.md D-019): pass the previous
                    # attempt's validation errors back in so the model fixes
                    # the specific mistake (most commonly a point duplicated
                    # across an ancestor and descendant tag) instead of this
                    # being an identical, blind repeat of the same request.
                    card = await build_literature_card(
                        state["review_question"], definitions, document, model,
                        previous_errors=attempt_errors or None,
                    )
                    store.write_json(
                        f"literature_cards/{document.paper_id}.json", _dump(card)
                    )
                audit = validate_literature_card(card, tree)
                if runtime.semantic_audits and audit.passed:
                    audit = await audit_literature_card_semantics(
                        card, definitions, model
                    )
                audit = audit.model_copy(update={"attempts": attempt})
                store.write_json(audit_path, _dump(audit))
                if audit.passed:
                    return {
                        "completed_card_paths": [
                            f"literature_cards/{document.paper_id}.json"
                        ]
                    }
                attempt_errors.append(
                    f"attempt {attempt} validation: " + "; ".join(audit.errors)
                )
            except Exception as error:  # worker isolation is an explicit requirement
                attempt_errors.append(
                    f"attempt {attempt} {type(error).__name__}: {error}"
                )
        failed_audit = CardAudit(
            paper_id=document.paper_id,
            passed=False,
            errors=attempt_errors,
            attempts=runtime.max_card_attempts,
        )
        store.write_json(audit_path, _dump(failed_audit))
        return {
            "failed_paper_ids": [document.paper_id],
            "errors": [
                f"Paper {document.paper_id} failed after "
                f"{runtime.max_card_attempts} attempts: "
                + "; ".join(attempt_errors)
            ],
        }

    async def create_tag_index(state: ReviewState) -> dict[str, object]:
        tree = _load_tree(store, state["tag_tree_path"])
        paths = _unique(state.get("completed_card_paths", []))
        cards = [LiteratureCard.model_validate(store.read_json(path)) for path in paths]
        index = build_tag_index(cards, tree)
        rendered = {
            tag_id: [item.model_dump(mode="json") for item in items]
            for tag_id, items in index.items()
        }
        path = store.write_json("tag_index.json", cast(JsonValue, rendered))
        return {"tag_index_path": path}

    async def analyze_coverage(state: ReviewState) -> dict[str, object]:
        tree = _load_tree(store, state["tag_tree_path"])
        index = _load_index(store, state["tag_index_path"])
        report = analyze_tag_coverage(index, tree)
        path = store.write_json("tag_coverage.json", _dump(report))
        warnings = [
            f"Insufficient evidence for tag {tag_id}."
            for tag_id in report.insufficient_tag_ids
            if tag_id != "TAG-ROOT"
        ]
        return {"coverage_report_path": path, "warnings": warnings}

    async def plan_sections(state: ReviewState) -> dict[str, object]:
        tree = _load_tree(store, state["tag_tree_path"])
        return {"writing_order": plan_writing_order(tree), "next_section_index": 0}

    def route_to_section(state: ReviewState) -> str:
        return "write_section" if state["writing_order"] else "write_review"

    async def write_next_section(state: ReviewState) -> dict[str, object]:
        position = state["next_section_index"]
        tag_id = state["writing_order"][position]
        draft_path = f"section_drafts/{tag_id}.json"
        summary_path = f"section_summaries/{tag_id}.json"
        audit_path = f"section_audits/{tag_id}.json"
        try:
            tree = _load_tree(store, state["tag_tree_path"])
            definitions = _load_definitions(store, state["tag_definitions_path"])
            definition = next(item for item in definitions if item.tag_id == tag_id)
            index = _load_index(store, state["tag_index_path"])
            node = next(item for item in tree if item.tag_id == tag_id)
            child_summaries = {
                child_id: SectionSummary.model_validate(
                    store.read_json(f"section_summaries/{child_id}.json")
                )
                for child_id in node.child_ids
                if store.exists(f"section_summaries/{child_id}.json")
            }
            # Per-role budgets (target/min/max), not a naive equal split —
            # see DECISIONS.md D-025/D-026 and word_budget.py. Keyed by the
            # tag's own title, which matches the outline headings
            # writing_prep.py::build_outline() produces (e.g. "Literature
            # Review"). Returns {} (no budget for anything) when no total
            # target was requested, exactly preserving pre-D-025 behavior.
            budgets = allocate_section_budgets(state.get("target_words"))
            budget = budgets.get(node.title)
            target_words = budget.target if budget else None
            min_words = budget.minimum if budget else None
            max_words = budget.maximum if budget else None
            context = build_section_context(
                tag_id,
                tree,
                definitions,
                index,
                child_summaries,
                target_words,
                output_language=state["output_language"],
                min_words=min_words,
                max_words=max_words,
            )
            store.write_json(f"section_contexts/{tag_id}.json", _dump(context))

            if definition.node_type == "container":
                summary_values = list(child_summaries.values())
                # A container tag has no prose of its own (assemble_review()
                # renders only its heading; its children carry the actual
                # content) — but SectionDraft.content requires min_length=1
                # (see writing/schemas.py). A literal "" here is an invalid
                # state that crashes deterministically on EVERY run with a
                # container tag, regardless of model behavior (see
                # DECISIONS.md D-020) — not a model-reliability issue at all,
                # which is why retrying it could never help. A single space
                # satisfies the schema while still rendering as empty:
                # assemble_review() checks `draft.content.strip()` before
                # emitting any body text for a section.
                draft = SectionDraft(
                    tag_id=tag_id, content=" ", cited_paper_ids=[], used_point_ids=[]
                )
                audit = SectionAudit(
                    tag_id=tag_id,
                    passed=True,
                    unsupported_claims=[],
                    out_of_scope_content=[],
                    invalid_paper_ids=[],
                    missing_key_points=[],
                    duplicated_child_content=[],
                    revision_instructions=[],
                )
                run_result = SectionRunResult(
                    draft=draft,
                    audits=[audit],
                    resolved=True,
                    unresolved_issues=[],
                )
                summary = SectionSummary(
                    tag_id=tag_id,
                    summary=" ".join(item.summary for item in summary_values),
                    cited_paper_ids=_unique(
                        [paper for item in summary_values for paper in item.cited_paper_ids]
                    ),
                    used_point_ids=_unique(
                        [point for item in summary_values for point in item.used_point_ids]
                    ),
                )
            else:
                section_errors: list[str] = []
                for section_attempt in range(1, runtime.max_section_attempts + 1):
                    try:
                        run_result = await write_section_with_revisions(
                            context,
                            model,
                            max_revision_rounds=runtime.max_section_revision_rounds,
                            semantic_audit=runtime.semantic_audits,
                        )
                        break
                    except Exception as error:
                        section_errors.append(
                            f"attempt {section_attempt} {type(error).__name__}: {error}"
                        )
                        if section_attempt == runtime.max_section_attempts:
                            raise RuntimeError("; ".join(section_errors)) from error
                draft = run_result.draft
                summary = SectionSummary(
                    tag_id=tag_id,
                    summary=draft.summary or draft.content[:1200],
                    cited_paper_ids=draft.cited_paper_ids,
                    used_point_ids=draft.used_point_ids,
                    evidence_limited=run_result.evidence_limited,
                )

            if not run_result.resolved:
                summary = SectionSummary(
                    tag_id=tag_id,
                    summary="Insufficient evidence: section audit was unresolved.",
                    cited_paper_ids=[],
                    used_point_ids=[],
                )

            persisted_draft = (
                draft
                if run_result.resolved
                else _unresolved_section_draft(tag_id, state["output_language"])
            )
            store.write_json(draft_path, _dump(persisted_draft))
            store.write_json(audit_path, _dump(run_result))
            store.write_json(summary_path, _dump(summary))
            result: dict[str, object] = {
                "completed_section_paths": [draft_path],
                "next_section_index": position + 1,
            }
            if not run_result.resolved:
                result["failed_tag_ids"] = [tag_id]
                result["warnings"] = [
                    f"Section {tag_id} has unresolved issues: "
                    + "; ".join(run_result.unresolved_issues)
                ]
            return result
        except Exception as error:
            draft = SectionDraft(
                tag_id=tag_id,
                content="*Section generation failed; no unsupported content was inserted.*",
                cited_paper_ids=[],
                used_point_ids=[],
            )
            summary = SectionSummary(
                tag_id=tag_id,
                summary=draft.content,
                cited_paper_ids=[],
                used_point_ids=[],
            )
            store.write_json(draft_path, _dump(draft))
            store.write_json(summary_path, _dump(summary))
            store.write_json(
                audit_path,
                {
                    "draft": draft.model_dump(mode="json"),
                    "audits": [],
                    "resolved": False,
                    "unresolved_issues": [f"{type(error).__name__}: {error}"],
                },
            )
            return {
                "completed_section_paths": [draft_path],
                "failed_tag_ids": [tag_id],
                "next_section_index": position + 1,
                "errors": [f"Section {tag_id} failed: {type(error).__name__}: {error}"],
            }

    def continue_sections(state: ReviewState) -> str:
        return (
            "write_section"
            if state["next_section_index"] < len(state["writing_order"])
            else "write_review"
        )

    async def write_review(state: ReviewState) -> dict[str, object]:
        tree = _load_tree(store, state["tag_tree_path"])
        definitions = _load_definitions(store, state["tag_definitions_path"])
        index = _load_index(store, state["tag_index_path"])
        cards = [
            LiteratureCard.model_validate(store.read_json(path))
            for path in _unique(state.get("completed_card_paths", []))
        ]
        metadata_by_paper_id = {card.paper_id: card.metadata for card in cards}
        drafts = {
            draft.tag_id: draft
            for path in _unique(state.get("completed_section_paths", []))
            for draft in [SectionDraft.model_validate(store.read_json(path))]
        }
        summaries: list[SectionSummary] = []
        resolved_tag_ids: set[str] = set()
        for tag_id in state["writing_order"]:
            summary_path = f"section_summaries/{tag_id}.json"
            audit_path = f"section_audits/{tag_id}.json"
            if not store.exists(summary_path) or not store.exists(audit_path):
                continue
            run_result = SectionRunResult.model_validate(store.read_json(audit_path))
            if run_result.resolved:
                resolved_tag_ids.add(tag_id)
                summaries.append(
                    SectionSummary.model_validate(store.read_json(summary_path))
                )
        summary_by_id = {summary.tag_id: summary for summary in summaries}
        top_level_ids = tree[0].child_ids
        top_level_summaries = [
            summary_by_id[tag_id] for tag_id in top_level_ids if tag_id in summary_by_id
        ]
        overview_points = direct_items("TAG-ROOT", index)
        for tag_id in top_level_ids:
            overview_points.extend(direct_items(tag_id, index))
        seen_points: set[str] = set()
        unique_overview_points: list[TagIndexItem] = []
        for item in overview_points:
            if item.point_id not in seen_points:
                seen_points.add(item.point_id)
                unique_overview_points.append(item)
        overview_points = unique_overview_points

        has_intro_evidence = bool(overview_points) or any(
            summary.used_point_ids for summary in summaries
        )
        has_conclusion_evidence = any(summary.used_point_ids for summary in summaries)
        bookend_budgets = allocate_section_budgets(state.get("target_words"))
        intro_budget = bookend_budgets.get("Introduction")
        conclusion_budget = bookend_budgets.get("Conclusion")
        if has_intro_evidence:
            introduction = await write_introduction(
                state["review_question"],
                tree,
                top_level_summaries,
                overview_points,
                model,
                output_language=state["output_language"],
                target_words=intro_budget.target if intro_budget else None,
                min_words=intro_budget.minimum if intro_budget else None,
                max_words=intro_budget.maximum if intro_budget else None,
            )
        else:
            chinese_output = state["output_language"].casefold().startswith("zh")
            introduction = SectionDraft(
                tag_id="INTRODUCTION",
                content=(
                    "*本综述可用证据不足。*"
                    if chinese_output
                    else "*Insufficient evidence is available for this review.*"
                ),
                cited_paper_ids=[],
                used_point_ids=[],
            )

        if has_conclusion_evidence:
            conclusion = await write_conclusion(
                state["review_question"],
                summaries,
                model,
                output_language=state["output_language"],
                target_words=conclusion_budget.target if conclusion_budget else None,
                min_words=conclusion_budget.minimum if conclusion_budget else None,
                max_words=conclusion_budget.maximum if conclusion_budget else None,
                # Only the REAL, already-generated Introduction — never the
                # "insufficient evidence" placeholder, which has nothing to
                # avoid repeating (see D-028 / review_writer.py docstring).
                introduction_content=introduction.content if has_intro_evidence else None,
            )
        else:
            chinese_output = state["output_language"].casefold().startswith("zh")
            conclusion = SectionDraft(
                tag_id="CONCLUSION",
                content=(
                    "*无法形成有证据支持的结论。*"
                    if chinese_output
                    else "*No evidence-backed conclusion can be drawn.*"
                ),
                cited_paper_ids=[],
                used_point_ids=[],
            )

        citation_lookup: dict[str, CitationInfo] = {}
        point_content_lookup: dict[str, str] = {}
        for items in index.values():
            for item in items:
                citation_lookup[item.paper_id] = item.citation
                point_content_lookup[item.point_id] = item.content

        intro_allowed_papers = _unique(
            [item.paper_id for item in overview_points]
            + [
                paper_id
                for summary in top_level_summaries
                for paper_id in summary.cited_paper_ids
            ]
        )
        intro_allowed_points = _unique(
            [item.point_id for item in overview_points]
            + [
                point_id
                for summary in top_level_summaries
                for point_id in summary.used_point_ids
            ]
        )
        body_allowed_papers = _unique(
            [paper_id for summary in summaries for paper_id in summary.cited_paper_ids]
        )
        body_allowed_points = _unique(
            [point_id for summary in summaries for point_id in summary.used_point_ids]
        )
        introduction_context = SectionWritingContext(
            tag_id="INTRODUCTION",
            title="Introduction",
            outline_path=[tree[0].title],
            depth=0,
            writing_mode="subtree",
            direct_points=overview_points,
            ancestor_context=[],
            child_summaries=[summary.summary for summary in top_level_summaries],
            allowed_paper_ids=intro_allowed_papers,
            allowed_point_ids=intro_allowed_points,
            citations={
                paper_id: citation_lookup[paper_id]
                for paper_id in intro_allowed_papers
                if paper_id in citation_lookup
            },
            sibling_titles=[],
            prohibited_topics=[],
        )
        conclusion_context = SectionWritingContext(
            tag_id="CONCLUSION",
            title="Conclusion",
            outline_path=[tree[0].title],
            depth=0,
            writing_mode="subtree",
            direct_points=[],
            ancestor_context=[],
            child_summaries=[summary.summary for summary in summaries],
            allowed_paper_ids=body_allowed_papers,
            allowed_point_ids=body_allowed_points,
            citations={
                paper_id: citation_lookup[paper_id]
                for paper_id in body_allowed_papers
                if paper_id in citation_lookup
            },
            sibling_titles=[],
            prohibited_topics=[],
        )
        bookend_warnings: list[str] = []
        if not audit_section(introduction, introduction_context).passed:
            bookend_warnings.append(
                "Introduction output violated its evidence context and was replaced."
            )
            introduction = SectionDraft(
                tag_id="INTRODUCTION",
                content="*Introduction generation did not satisfy evidence constraints.*",
                cited_paper_ids=[],
                used_point_ids=[],
            )
        if not audit_section(conclusion, conclusion_context).passed:
            bookend_warnings.append(
                "Conclusion output violated its evidence context and was replaced."
            )
            conclusion = SectionDraft(
                tag_id="CONCLUSION",
                content="*No evidence-backed conclusion can be drawn.*",
                cited_paper_ids=[],
                used_point_ids=[],
            )

        async def assemble_and_audit() -> tuple[ReviewDocument, FullReviewAudit]:
            current_review = assemble_review(
                tree,
                definitions,
                drafts,
                introduction,
                conclusion,
                citation_lookup,
                citation_style=state["citation_style"],
                metadata_by_paper_id=metadata_by_paper_id,
                output_language=state["output_language"],
            )
            rendered_citations = build_citation_map(
                current_review.cited_paper_ids,
                metadata_by_paper_id,
                state["citation_style"],
            )
            current_audit = audit_full_review(
                current_review,
                tree,
                definitions,
                drafts,
                introduction,
                conclusion,
                rendered_citations,
                citation_style=state["citation_style"],
            )
            if runtime.semantic_audits and current_audit.passed:
                semantic = await audit_full_review_semantics(
                    state["review_question"],
                    current_review,
                    summaries,
                    model,
                    section_contents={
                        tag_id: draft.content for tag_id, draft in drafts.items()
                    },
                )
                if semantic.visual_recommendations:
                    try:
                        approved_visuals = await audit_visual_recommendation_evidence(
                            semantic.visual_recommendations,
                            point_content_lookup,
                            model,
                        )
                    except Exception:
                        approved_visuals = []
                        bookend_warnings.append(
                            "Visual recommendation evidence audit failed; suggestions omitted."
                        )
                    semantic = semantic.model_copy(
                        update={"visual_recommendations": approved_visuals}
                    )
                if semantic.visual_recommendations:
                    current_review = assemble_review(
                        tree,
                        definitions,
                        drafts,
                        introduction,
                        conclusion,
                        citation_lookup,
                        citation_style=state["citation_style"],
                        metadata_by_paper_id=metadata_by_paper_id,
                        output_language=state["output_language"],
                        visual_recommendations=semantic.visual_recommendations,
                    )
                    current_audit = audit_full_review(
                        current_review,
                        tree,
                        definitions,
                        drafts,
                        introduction,
                        conclusion,
                        build_citation_map(
                            current_review.cited_paper_ids,
                            metadata_by_paper_id,
                            state["citation_style"],
                        ),
                        citation_style=state["citation_style"],
                    )
                current_audit = current_audit.model_copy(
                    update={
                        "passed": current_audit.passed and semantic.passed,
                        "semantic_checked": True,
                        "semantic_issues": semantic.issues,
                        "revision_instructions": [
                            *current_audit.revision_instructions,
                            *semantic.revision_instructions,
                        ],
                        "visual_recommendations": semantic.visual_recommendations,
                    }
                )
            return current_review, current_audit

        review, audit = await assemble_and_audit()
        definition_by_id = {item.tag_id: item for item in definitions}
        editable_drafts = {
            tag_id: draft
            for tag_id, draft in drafts.items()
            if tag_id in resolved_tag_ids
            and definition_by_id[tag_id].node_type != "container"
        }
        contexts = {
            tag_id: SectionWritingContext.model_validate(
                store.read_json(f"section_contexts/{tag_id}.json")
            )
            for tag_id in editable_drafts
        }
        full_revision_round = 0
        while not audit.passed and full_revision_round < runtime.max_full_revision_rounds:
            try:
                revised_introduction, revised_sections, revised_conclusion = (
                    await revise_full_review(
                        introduction,
                        editable_drafts,
                        conclusion,
                        contexts,
                        audit,
                        model,
                    )
                )
            except Exception as error:
                audit = audit.model_copy(
                    update={
                        "passed": False,
                        "errors": [
                            *audit.errors,
                            f"Full revision failed: {type(error).__name__}: {error}",
                        ],
                        "revision_instructions": [
                            *audit.revision_instructions,
                            "Retry the bounded full-review revision.",
                        ],
                    }
                )
                full_revision_round += 1
                continue
            section_audits: dict[str, SectionAudit] = {}
            for tag_id, revised_draft in revised_sections.items():
                section_audit = audit_section(revised_draft, contexts[tag_id])
                if runtime.semantic_audits and section_audit.passed:
                    section_audit = await audit_section_semantics(
                        revised_draft, contexts[tag_id], model
                    )
                section_audits[tag_id] = section_audit
            invalid_sections = [
                tag_id
                for tag_id, section_audit in section_audits.items()
                if not section_audit.passed
            ]
            if not audit_section(revised_introduction, introduction_context).passed:
                invalid_sections.append("INTRODUCTION")
            if not audit_section(revised_conclusion, conclusion_context).passed:
                invalid_sections.append("CONCLUSION")
            if invalid_sections:
                audit = audit.model_copy(
                    update={
                        "passed": False,
                        "errors": [
                            *audit.errors,
                            "Full revision violated original section contexts: "
                            + ", ".join(invalid_sections),
                        ],
                        "revision_instructions": [
                            *audit.revision_instructions,
                            "Keep every revised section inside its original evidence context.",
                        ],
                    }
                )
                full_revision_round += 1
                continue

            introduction = revised_introduction
            conclusion = revised_conclusion
            editable_drafts = revised_sections
            drafts.update(revised_sections)
            revised_summary_by_id = {
                summary.tag_id: summary for summary in summaries
            }
            for tag_id, revised_draft in revised_sections.items():
                revised_summary_by_id[tag_id] = SectionSummary(
                    tag_id=tag_id,
                    summary=revised_draft.summary or revised_draft.content[:1200],
                    cited_paper_ids=revised_draft.cited_paper_ids,
                    used_point_ids=revised_draft.used_point_ids,
                )
                store.write_json(f"section_drafts/{tag_id}.json", _dump(revised_draft))
                store.write_json(
                    f"section_audits/{tag_id}.json",
                    {
                        "draft": revised_draft.model_dump(mode="json"),
                        "audits": [section_audits[tag_id].model_dump(mode="json")],
                        "resolved": True,
                        "unresolved_issues": [],
                    },
                )
                store.write_json(
                    f"section_summaries/{tag_id}.json",
                    _dump(revised_summary_by_id[tag_id]),
                )
            summaries = [
                revised_summary_by_id.get(summary.tag_id, summary) for summary in summaries
            ]
            review, audit = await assemble_and_audit()
            full_revision_round += 1

        introduction_path = store.write_json(
            "section_drafts/INTRODUCTION.json", _dump(introduction)
        )
        conclusion_path = store.write_json(
            "section_drafts/CONCLUSION.json", _dump(conclusion)
        )
        final_path = store.write_text("final_review.md", review.markdown)
        store.write_json(
            "references.json",
            cast(JsonValue, {
                "citation_style": state["citation_style"],
                "cited_paper_ids": review.cited_paper_ids,
                "references": review.references,
            }),
        )
        audit_path = store.write_json("full_review_audit.json", _dump(audit))
        visual_recommendations_path = store.write_json(
            "visual_recommendations.json",
            _dump_many(audit.visual_recommendations),
        )
        errors = [] if audit.passed else [
            "Full review audit failed: " + "; ".join(audit.revision_instructions)
        ]
        return {
            "introduction_path": introduction_path,
            "conclusion_path": conclusion_path,
            "final_review_path": final_path,
            "full_review_audit_path": audit_path,
            "visual_recommendations_path": visual_recommendations_path,
            "warnings": bookend_warnings,
            "errors": errors,
        }

    builder = StateGraph(ReviewState)
    builder.add_node("initialize", initialize)
    builder.add_node("define_tag_semantics", define_tags)
    builder.add_node("discover_documents", register_documents)
    builder.add_node("build_literature_card", build_card_worker)
    builder.add_node("build_tag_index", create_tag_index)
    builder.add_node("analyze_tag_coverage", analyze_coverage)
    builder.add_node("plan_writing_order", plan_sections)
    builder.add_node("write_section", write_next_section)
    builder.add_node("write_review", write_review)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "define_tag_semantics")
    builder.add_edge("define_tag_semantics", "discover_documents")
    builder.add_conditional_edges("discover_documents", dispatch_cards)
    builder.add_edge("build_literature_card", "build_tag_index")
    builder.add_edge("build_tag_index", "analyze_tag_coverage")
    builder.add_edge("analyze_tag_coverage", "plan_writing_order")
    builder.add_conditional_edges("plan_writing_order", route_to_section)
    builder.add_conditional_edges("write_section", continue_sections)
    builder.add_edge("write_review", END)
    return builder.compile(checkpointer=checkpointer)


async def run_review_async(
    request: ReviewInput,
    model: LanguageModel | None = None,
    *,
    config: ReviewConfig | None = None,
    run_id: str | None = None,
    document_parser: DocumentParser | None = None,
    outline_base_directory: str | Path | None = None,
) -> ReviewResult:
    """Run or resume an outline-driven review with a persistent SQLite checkpoint."""

    runtime = config or ReviewConfig()
    actual_run_id = run_id or uuid4().hex
    if not _RUN_ID.fullmatch(actual_run_id) or ".." in actual_run_id:
        raise ValueError("run_id contains unsafe characters")
    run_directory = (Path(request.output_directory) / actual_run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    store = LocalArtifactStore(run_directory)
    resolved_outline = resolve_outline_source(
        request.outline, base_directory=outline_base_directory
    )
    outline_snapshot: JsonValue = {
        "content": resolved_outline.content,
        "content_sha256": hashlib.sha256(
            resolved_outline.content.encode()
        ).hexdigest(),
        "source_path": resolved_outline.source_path,
    }
    request_payload = cast(JsonValue, request.model_dump(mode="json"))
    if store.exists("input.json") and store.read_json("input.json") != request_payload:
        raise ValueError(
            "run_id already exists with different inputs; use a new run_id"
        )
    store.write_json("input.json", request_payload)
    if (
        store.exists("outline_source.json")
        and store.read_json("outline_source.json") != outline_snapshot
    ):
        raise ValueError(
            "run_id already exists with a different resolved outline; use a new run_id"
        )
    store.write_json("outline_source.json", outline_snapshot)
    prepared_literature_directory = Path(request.literature_directory).resolve()
    preparation_warnings: tuple[str, ...] = ()
    if contains_pdf(prepared_literature_directory):
        if document_parser is None:
            from writing.adapters.mineru import MinerUDocumentParser

            auto_parser = MinerUDocumentParser.from_environment()
            try:
                preparation = await auto_parser.prepare_directory(
                    prepared_literature_directory,
                    run_directory / "parsed_literature",
                )
            finally:
                await auto_parser.aclose()
        else:
            preparation = await document_parser.prepare_directory(
                prepared_literature_directory,
                run_directory / "parsed_literature",
            )
        prepared_literature_directory = preparation.directory
        preparation_warnings = preparation.warnings
    checkpoint_path = run_directory / "checkpoints.sqlite"
    initial: ReviewState = {
        "run_id": actual_run_id,
        "review_question": request.review_question,
        "outline_source": resolved_outline.content,
        "literature_directory": str(prepared_literature_directory),
        "output_directory": request.output_directory,
        "output_language": request.output_language,
        "citation_style": request.citation_style,
        "target_words": request.target_words,
        "completed_card_paths": _reset_reduced_list(),
        "failed_paper_ids": _reset_reduced_list(),
        "completed_section_paths": _reset_reduced_list(),
        "failed_tag_ids": _reset_reduced_list(),
        "warnings": _reset_reduced_list(preparation_warnings),
        "errors": _reset_reduced_list(),
    }
    owns_model = model is None
    resolved_model = resolve_language_model(model)
    try:
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
            graph = build_review_graph(resolved_model, store, runtime, checkpointer)
            final = cast(
                ReviewState,
                await graph.ainvoke(
                    initial,
                    {
                        "configurable": {"thread_id": actual_run_id},
                        "recursion_limit": runtime.recursion_limit,
                        "max_concurrency": runtime.max_concurrency,
                    },
                ),
            )
    finally:
        if owns_model:
            await close_language_model(resolved_model)

    audit_passed = False
    if final.get("full_review_audit_path"):
        audit_passed = FullReviewAudit.model_validate(
            store.read_json(final["full_review_audit_path"])
        ).passed
    relative_final = final.get("final_review_path")
    return ReviewResult(
        run_id=actual_run_id,
        succeeded=bool(relative_final and audit_passed and not final.get("failed_tag_ids")),
        run_directory=str(run_directory),
        final_review_path=(
            str(run_directory / relative_final) if relative_final is not None else None
        ),
        warnings=_unique(final.get("warnings", [])),
        errors=_unique(final.get("errors", [])),
        failed_paper_ids=_unique(final.get("failed_paper_ids", [])),
        failed_tag_ids=_unique(final.get("failed_tag_ids", [])),
    )


def run_review(
    request: ReviewInput,
    model: LanguageModel | None = None,
    *,
    config: ReviewConfig | None = None,
    run_id: str | None = None,
    document_parser: DocumentParser | None = None,
    outline_base_directory: str | Path | None = None,
) -> ReviewResult:
    """Synchronous wrapper for environments without an active event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_review_async(
                request,
                model,
                config=config,
                run_id=run_id,
                document_parser=document_parser,
                outline_base_directory=outline_base_directory,
            )
        )
    raise RuntimeError("run_review cannot run inside an event loop; await run_review_async")

```

### `dev_agent/agent.py`

```python
"""Dev Agent (#25): ticket -> plan -> failing test -> patch -> green, then a
diff for a human to approve.

A LangGraph state graph reads a ticket, plans the files to change and the
acceptance test, writes the test, runs it (it must fail), then patches and
runs the tests again, feeding pytest's output back each time, for at most
MAX_ROUNDS patch rounds. It works in a detached git worktree, writes only to
the paths in its plan (the patch may not touch its own test), runs pytest in
a subprocess with a timeout, calls the model only through
shared/utilities/llm_provider.py, and logs every step to an audit trail.

A patch is a list of search/replace edits, never a retyped file: a small
local model copying a whole file back inside a JSON string corrupts lines
it never meant to touch (a regex's backslashes, most often). Each round the
model also sees the diff of its own changes so far and any edit that didn't
apply, and the existing tests that use the changed files run beside the
acceptance test, so a broken unrelated line can't come back green.

It never commits, pushes or merges: every git call goes through `git()`,
which only allows the read-only and worktree commands listed below. The
result is a patch file the human applies with `git apply` after review.

    researchgenie-dev "Add subtract(a, b) to calc.py"
    researchgenie-dev --spec ticket.md [--repo PATH] [--timeout SECONDS]
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.utilities import llm_provider  # noqa: E402
from shared.utilities.run_paths import slugify  # noqa: E402

MAX_ROUNDS = 3
TEST_TIMEOUT = 300.0     # seconds per pytest run
OUTPUT_TAIL = 4000       # characters of pytest output fed back to the model
REPO_LISTING = 400       # files of the repo shown to the planner
MAX_TOKENS = 16000       # a patch round's answer; thinking is off so all of it is the answer
GUARD_TESTS = 10         # existing tests that use the changed files, run beside the new one
DIFF_TAIL = 6000         # characters of "your changes so far" shown each round
_PATH_CHARS = re.compile(r"[\w.\-/]+")  # what a repo path may contain; a garbled one is refused
# git subcommands the agent may run; anything else (commit, push, merge,
# rebase, reset, checkout, ...) is refused before a process starts.
GIT_ALLOWED = {"worktree": {"add", "remove", "prune"}, "add": {"--intent-to-add"},
               "diff": None, "ls-files": None, "rev-parse": None, "status": None}

Complete = Callable[..., Awaitable[llm_provider.Completion]]


class DevAgentError(RuntimeError):
    """A refused git command, or a plan the agent cannot act on."""


def git(repo: Path, *args: str) -> str:
    """Run an allow-listed git command in `repo` and return its stdout."""
    rest = list(args)
    while rest and rest[0] == "-c":  # `-c key=value` pairs before the subcommand
        rest = rest[2:]
    command = rest[0] if rest else ""
    allowed = GIT_ALLOWED.get(command, ())
    if command not in GIT_ALLOWED or (allowed and not set(rest[1:2]) & allowed):
        raise DevAgentError(f"The Dev Agent never runs `git {' '.join(args)}`.")
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if done.returncode:
        raise DevAgentError(f"git {command} failed: {done.stderr.strip()}")
    return done.stdout


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "files": {"type": "array", "items": {"type": "string"},
                  "description": "Repo-relative paths the patch will create or change."},
        "test_file": {"type": "string",
                      "description": "Repo-relative path of the new acceptance test file."},
    },
    "required": ["summary", "files", "test_file"],
}
FILES_SCHEMA = {
    "type": "object",
    "properties": {"files": {"type": "array", "items": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}}},
    "required": ["files"],
}
EDITS_SCHEMA = {
    "type": "object",
    "properties": {"edits": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "Text copied exactly from the current file, "
                    "with enough lines to occur only once. Empty only to create a new file."},
            "new": {"type": "string", "description": "The text that replaces it."}},
        "required": ["path", "old", "new"]}}},
    "required": ["edits"],
}
SYSTEM = ("You are a careful software engineer working test-first on a small, well-specified "
          "ticket. Change as little as possible. Answer only with the requested JSON. "
          "Change existing files with search/replace edits: copy only the lines you change, plus "
          "enough context to be unique, exactly as they are in the file. Never retype a whole file.")


class State(TypedDict, total=False):
    ticket: str
    worktree: str
    plan: dict
    guards: list[str]   # existing tests that use the changed files and passed before the patch
    passed: bool
    output: str
    problems: list[str]  # last round's edits that were not applied, and why
    rounds: int
    status: str


def _safe_path(path: str) -> str | None:
    """The normalized repo-relative POSIX path, or None if it escapes the repo."""
    pure = PurePosixPath(path.strip().replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts or ":" in pure.parts[0] \
            or not _PATH_CHARS.fullmatch(str(pure)):
        return None  # escapes the repo, or garbled (quotes, braces, spaces from broken JSON)
    return str(pure)


class _Run:
    """One agent run: its worktree, its audit trail, and the model."""

    def __init__(self, repo: Path, audit_dir: Path, complete: Complete, test_timeout: float):
        self.repo, self.audit_dir = repo, audit_dir
        self.complete, self.test_timeout = complete, test_timeout
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def log(self, step: str, **fields: Any) -> None:
        entry = {"step": step, "at": datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}
        with (self.audit_dir / "audit.jsonl").open("a", encoding="utf-8") as audit:
            audit.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def ask(self, user: str, schema: dict) -> dict:
        comp = await self.complete(system=SYSTEM, user=user, schema=schema,
                                   max_tokens=MAX_TOKENS, think=False)
        if comp.stop != "end" or not comp.text.strip():
            raise DevAgentError(f"The model gave no usable answer (stop: {comp.stop}).")
        try:
            return json.loads(comp.text)
        except json.JSONDecodeError:
            raise DevAgentError("The model's answer was not valid JSON.") from None

    def read(self, worktree: Path, path: str) -> str:
        file = worktree / path
        return file.read_text(encoding="utf-8") if file.is_file() else "(new file)"

    def write(self, worktree: Path, allowed: set[str], files: list[dict], step: str) -> list[str]:
        """Writes each file whose path is in `allowed`; refuses (and logs) the rest."""
        written = []
        for item in files:
            path = _safe_path(item.get("path", ""))
            if path is None or path not in allowed:
                self.log("write_refused", path=item.get("path"), during=step,
                         allowed=sorted(allowed))
                continue
            target = worktree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item.get("content", ""), encoding="utf-8", newline="")
            written.append(path)
        self.log("write", during=step, paths=written)
        return written

    def edit(self, worktree: Path, allowed: set[str], edits: list[dict], step: str) -> list[str]:
        """Applies each search/replace edit to a path in `allowed` whose `old`
        text occurs exactly once (an empty `old` creates a new file). Returns
        the problems with the edits that were not applied, for the next round.
        The file keeps its own line endings."""
        # ponytail: exact match only; add a whitespace-tolerant match if models keep missing on indentation.
        applied, problems = [], []
        for item in edits:
            path = _safe_path(str(item.get("path", "")))
            if path is None or path not in allowed:
                self.log("write_refused", path=item.get("path"), during=step, allowed=sorted(allowed))
                continue
            old = str(item.get("old", "")).replace("\r\n", "\n")
            new = str(item.get("new", "")).replace("\r\n", "\n")
            target = worktree / path
            if not old:
                if target.exists():
                    problems.append(f"{path}: `old` was empty but the file exists; copy the lines "
                                    "to replace into `old`.")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(new, encoding="utf-8", newline="")
                applied.append(path)
                continue
            raw = target.read_bytes().decode("utf-8") if target.is_file() else ""
            text = raw.replace("\r\n", "\n")
            found = text.count(old)
            if found != 1:
                problems.append(f"{path}: this `old` text occurs {found} time(s) in the file, not "
                                "once, so the edit was not applied. Copy it exactly from the "
                                f"current file, with more lines if needed:\n{old[:400]}")
                continue
            text = text.replace(old, new, 1)
            target.write_bytes((text.replace("\n", "\r\n") if "\r\n" in raw else text).encode("utf-8"))
            applied.append(path)
        self.log("edit", during=step, paths=applied, problems=problems)
        return problems

    def guard_tests(self, worktree: Path, plan: dict) -> list[str]:
        """Existing test files that mention a changed module by name."""
        stems = {PurePosixPath(f).stem for f in plan["files"] if f.endswith(".py")} - {"__init__"}
        found = []
        for path in git(worktree, "ls-files").splitlines():
            name = PurePosixPath(path).name
            if path == plan["test_file"] or not (name.startswith("test_") and name.endswith(".py")):
                continue
            text = (worktree / path).read_text(encoding="utf-8", errors="replace")
            if any(re.search(rf"\b{re.escape(stem)}\b", text) for stem in stems):
                found.append(path)
        return found[:GUARD_TESTS]

    def example_test(self, worktree: Path, test_file: str) -> str:
        """The setup and imports of an existing test beside the new one, so the
        model copies this repo's real import pattern (hyphenated service folders
        are put on sys.path, not imported as packages)."""
        folder = str(PurePosixPath(test_file).parent)
        siblings = sorted(p for p in git(worktree, "ls-files").splitlines()
                          if str(PurePosixPath(p).parent) == folder and p != test_file
                          and PurePosixPath(p).name.startswith("test_") and p.endswith(".py"))
        if not siblings:
            return ""
        head = []
        for line in (worktree / siblings[0]).read_text(encoding="utf-8").splitlines()[:60]:
            if line.startswith(("def ", "class ", "@")):
                break
            head.append(line)
        return f"--- {siblings[0]} (its imports and setup) ---\n" + "\n".join(head).strip()

    async def pytest(self, worktree: Path, *test_files: str) -> tuple[bool, str]:
        # ponytail: the acceptance test and its guards only; the human's review and CI run the full suite.
        # -B: no .pyc between rounds. A same-size patch written within the same second
        # as the last one would otherwise run the previous round's cached bytecode.
        command = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *test_files]
        try:
            done = await asyncio.to_thread(subprocess.run, command, cwd=worktree,
                                           capture_output=True, text=True,
                                           timeout=self.test_timeout)
        except subprocess.TimeoutExpired:
            return False, f"pytest timed out after {self.test_timeout:.0f} s."
        return done.returncode == 0, (done.stdout + done.stderr)[-OUTPUT_TAIL:]


def _build_graph(run: _Run):
    async def plan(state: State) -> dict:
        worktree = Path(state["worktree"])
        listing = git(worktree, "ls-files").splitlines()[:REPO_LISTING]
        plan = await run.ask(f"Ticket:\n{state['ticket']}\n\nRepository files:\n"
                             + "\n".join(listing)
                             + "\n\nPlan the smallest change: the files to create or edit, "
                               "and one new pytest file that fails until the ticket is done.",
                             PLAN_SCHEMA)
        test_file = _safe_path(str(plan.get("test_file", "")))
        files = [_safe_path(str(f)) for f in plan.get("files", [])]
        files = list(dict.fromkeys(f for f in files if f != test_file))  # the test is written separately
        if not files or None in files or test_file is None:
            raise DevAgentError(f"The plan is unusable: {plan}")
        if (worktree / test_file).exists():
            # Writing it would replace an existing test file with the new acceptance test.
            raise DevAgentError(f"The plan's test file {test_file} already exists; the acceptance "
                                "test must be a new file.")
        plan = {**plan, "files": files, "test_file": test_file}
        run.log("plan", plan=plan)
        return {"plan": plan, "rounds": 0, "problems": []}

    async def write_test(state: State) -> dict:
        worktree, plan = Path(state["worktree"]), state["plan"]
        sources = "\n\n".join(f"--- {f} ---\n{run.read(worktree, f)}" for f in plan["files"])
        example = run.example_test(worktree, plan["test_file"])
        reply = await run.ask(f"Ticket:\n{state['ticket']}\n\nPlan: {plan.get('summary', '')}\n\n"
                              f"Current files:\n{sources}\n\n"
                              + (f"An existing test in the same folder; import the code the same "
                                 f"way:\n{example}\n\n" if example else "")
                              + f"Write the acceptance test {plan['test_file']}, as a complete new "
                              "file. It must fail now and pass once the ticket is done.", FILES_SCHEMA)
        run.write(worktree, {plan["test_file"]}, reply.get("files", []), "write_test")
        return {}

    async def red(state: State) -> dict:
        """The acceptance test must fail now. The guards are kept only if they
        pass now: an existing test that already fails can't vouch for the patch."""
        worktree = Path(state["worktree"])
        passed, output = await run.pytest(worktree, state["plan"]["test_file"])
        run.log("tests", round=0, passed=passed, output=output)
        guards = []
        for guard in run.guard_tests(worktree, state["plan"]):
            ok, _ = await run.pytest(worktree, guard)
            if ok:
                guards.append(guard)
            else:
                run.log("guard_dropped", test=guard)
        run.log("guards", tests=guards)
        return {"passed": passed, "output": output, "guards": guards}

    async def run_tests(state: State) -> dict:
        passed, output = await run.pytest(Path(state["worktree"]), state["plan"]["test_file"],
                                          *state.get("guards", []))
        run.log("tests", round=state["rounds"], passed=passed, output=output)
        return {"passed": passed, "output": output}

    async def patch(state: State) -> dict:
        worktree, plan = Path(state["worktree"]), state["plan"]
        n = state["rounds"] + 1
        sources = "\n\n".join(f"--- {f} ---\n{run.read(worktree, f)}" for f in plan["files"])
        diff = git(worktree, "diff", "--", *plan["files"])[-DIFF_TAIL:]
        guards = state.get("guards", [])
        problems = "\n".join(f"- {p}" for p in state.get("problems", []))
        reply = await run.ask(
            f"Patch round {n} of {MAX_ROUNDS}.\n\nTicket:\n{state['ticket']}\n\n"
            f"Acceptance test ({plan['test_file']}, do not change it):\n"
            f"{run.read(worktree, plan['test_file'])}\n\n"
            f"Files you may change (current content):\n{sources}\n\n"
            f"Your changes so far (git diff against the original):\n{diff or '(none yet)'}\n"
            "Undo any change in it that the ticket doesn't need.\n\n"
            + (f"Edits from the last round that were not applied:\n{problems}\n\n" if problems else "")
            + "Last pytest output (the acceptance test"
            + (f", and existing tests that use these files and must keep passing: {', '.join(guards)}"
               if guards else "") + f"):\n{state['output']}\n\n"
            "Return search/replace edits.", EDITS_SCHEMA)
        problems = run.edit(worktree, set(plan["files"]), reply.get("edits", []), f"patch_{n}")
        return {"rounds": n, "problems": problems}

    def after_red(state: State) -> str:
        return "finish" if state["passed"] else "patch"  # a test that passes first proves nothing

    def after_tests(state: State) -> str:
        return "finish" if state["passed"] or state["rounds"] >= MAX_ROUNDS else "patch"

    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("write_test", write_test)
    graph.add_node("red", red)
    graph.add_node("patch", patch)
    graph.add_node("tests", run_tests)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "write_test")
    graph.add_edge("write_test", "red")
    graph.add_conditional_edges("red", after_red, {"patch": "patch", "finish": END})
    graph.add_edge("patch", "tests")
    graph.add_conditional_edges("tests", after_tests, {"patch": "patch", "finish": END})
    return graph.compile()


def _report(state: State, diff: str, audit_dir: Path) -> dict:
    rounds, passed = state.get("rounds", 0), state.get("passed", False)
    if passed and rounds:
        status = "green"
        guards = state.get("guards", [])
        summary = (f"The acceptance test is green after {rounds} patch round(s)"
                   + (f", and {len(guards)} existing test file(s) that use the changed files still "
                      "pass" if guards else "") + "; the rest of the suite was not run. Review "
                   "patch.diff, run the full tests, then `git apply` it.")
    elif passed:
        status = "not_red"
        summary = ("The new test passed before any patch, so it does not test the ticket. "
                   "Nothing was patched.")
    else:
        status = "failed"
        summary = (f"Stopped after {rounds} patch round(s): the test did not reach green. "
                   f"Last pytest output:\n{state.get('output', '')}")
    return {"status": status, "rounds": rounds, "summary": summary, "diff": diff,
            "audit_dir": str(audit_dir)}


async def run_dev_agent(ticket: str, *, repo: Path | str = ROOT, complete: Complete | None = None,
                        audit_root: Path | None = None,
                        test_timeout: float = TEST_TIMEOUT) -> dict:
    """Runs the agent on `ticket` in a throwaway worktree of `repo` and
    returns {status, rounds, summary, diff, audit_dir}. status is "green",
    "failed" or "not_red"; the diff is also saved as <audit_dir>/patch.diff."""
    repo = Path(git(Path(repo), "rev-parse", "--show-toplevel").strip())
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    audit_dir = (audit_root or ROOT / "outputs" / "dev_agent") / f"{stamp}_{slugify(ticket[:60])}"
    run = _Run(repo, audit_dir, complete or llm_provider.complete_json, test_timeout)
    worktree = Path(tempfile.mkdtemp(prefix="dev-agent-")) / "worktree"
    git(repo, "worktree", "add", "--detach", str(worktree), "HEAD")
    run.log("start", ticket=ticket, repo=str(repo), worktree=str(worktree))
    state: State = {"ticket": ticket, "worktree": str(worktree), "rounds": 0}
    try:
        try:
            state = await _build_graph(run).ainvoke(state)
        except DevAgentError as error:
            run.log("error", message=str(error))
            state = {**state, "passed": False, "output": str(error)}
        plan = state.get("plan") or {}
        paths = [p for p in [*plan.get("files", []), plan.get("test_file")] if p]
        if paths:
            git(worktree, "add", "--intent-to-add", "--", *paths)  # so new files show in the diff
        diff = git(worktree, "diff", "--", *paths) if paths else ""
    finally:
        git(repo, "worktree", "remove", "--force", str(worktree))
    report = _report(state, diff, audit_dir)
    (audit_dir / "patch.diff").write_text(diff, encoding="utf-8")
    (audit_dir / "report.md").write_text(f"# Dev Agent report\n\n{ticket}\n\n**{report['status']}**"
                                         f" — {report['summary']}\n", encoding="utf-8")
    run.log("finish", status=report["status"], rounds=report["rounds"])
    return report


def read_ticket(args: list[str]) -> str:
    """The ticket from `--spec FILE`, or the remaining arguments as text."""
    if "--spec" in args:
        return Path(args[args.index("--spec") + 1]).read_text(encoding="utf-8")
    return " ".join(args).strip()


def _option(args: list[str], name: str, default: str | None = None) -> tuple[str | None, list[str]]:
    if name not in args:
        return default, args
    i = args.index(name)
    return args[i + 1], args[:i] + args[i + 2:]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    repo, args = _option(args, "--repo", str(ROOT))
    timeout, args = _option(args, "--timeout", str(TEST_TIMEOUT))
    ticket = read_ticket(args)
    if not ticket:
        print(__doc__.split("\n\n")[-1])
        sys.exit(2)
    report = asyncio.run(run_dev_agent(ticket, repo=repo, test_timeout=float(timeout)))
    print(report["summary"])
    print(f"\nAudit trail and patch: {report['audit_dir']}")
    if report["diff"]:
        print("\n" + report["diff"])
    sys.exit(0 if report["status"] == "green" else 1)


if __name__ == "__main__":
    main()

```

### Real tests for this lab

### `tests/dev_agent/test_dev_agent.py`

```python
"""The Dev Agent (#25) on a throwaway fixture repo, with a scripted model.

Every test runs the real graph: a real git worktree, real file writes
through the path guard, and a real pytest subprocess. Patches are
search/replace edits; the existing tests that use a changed file guard it.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev_agent import agent  # noqa: E402
from shared.utilities.llm_provider import Completion  # noqa: E402

TICKET = "Add subtract(a, b) to calc.py; it returns a - b."
TEST = "from calc import subtract\n\n\ndef test_subtract():\n    assert subtract(5, 3) == 2\n"
BASE = "def add(a, b):\n    return a + b\n"
WRONG = "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a + b\n"
RIGHT = "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n"
GUARD = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def _edits(*edits):
    return {"edits": [{"path": path, "old": old, "new": new} for path, old, new in edits]}


ADD_WRONG = _edits(("calc.py", BASE, WRONG))
ADD_RIGHT = _edits(("calc.py", BASE, RIGHT))
FIX_WRONG = _edits(("calc.py", "def subtract(a, b):\n    return a + b\n",
                    "def subtract(a, b):\n    return a - b\n"))


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "fixture-repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "calc.py").write_text(BASE, encoding="utf-8")
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_add.py").write_text(  # an existing test that uses calc.py
        "import sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path(__file__).parents[1]))\n"
        + GUARD, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@example.org", "-c", "user.name=t", "add", "-A")
    _git(repo, "-c", "user.email=t@example.org", "-c", "user.name=t", "commit", "-qm", "base")
    return repo


PLAN = {"summary": "add subtract", "files": ["calc.py"], "test_file": "tests/test_calc.py"}


def _model(*patches, plan=PLAN):
    """Answers the plan, then the test, then each patch round in turn (the
    last patch repeats). Records every prompt it was sent, and its options."""
    prompts, options = [], []
    replies = iter([plan, {"files": [{"path": "tests/test_calc.py", "content": TEST}]}])

    async def complete(*, system, user, schema, **kwargs):
        prompts.append(user)
        options.append(kwargs)
        reply = next(replies, None)
        if reply is None:
            patch_round = sum("Patch round" in p for p in prompts) - 1
            reply = patches[min(patch_round, len(patches) - 1)]
        return Completion(text=json.dumps(reply))

    complete.prompts, complete.options = prompts, options
    return complete


def _run(repo, tmp_path, model):
    return asyncio.run(agent.run_dev_agent(TICKET, repo=repo, complete=model,
                                           audit_root=tmp_path / "audit", test_timeout=120))


def _audit(report):
    return [json.loads(line) for line in
            (Path(report["audit_dir"]) / "audit.jsonl").read_text(encoding="utf-8").splitlines()]


def test_reaches_green_within_three_rounds_and_the_audit_shows_red_then_green(repo, tmp_path):
    model = _model(ADD_WRONG, FIX_WRONG)
    report = _run(repo, tmp_path, model)

    assert report["status"] == "green" and report["rounds"] == 2
    runs = [e["passed"] for e in _audit(report) if e["step"] == "tests"]
    assert runs == [False, False, True]  # red before the patch, red, then green
    assert "return a - b" in report["diff"] and "def test_subtract" in report["diff"]
    assert "AssertionError" in model.prompts[-1] or "assert" in model.prompts[-1]  # pytest output fed back
    assert (Path(report["audit_dir"]) / "patch.diff").read_text(encoding="utf-8") == report["diff"]
    # The real checkout is untouched; the work happened in a worktree.
    assert "subtract" not in (repo / "calc.py").read_text(encoding="utf-8")


def test_a_write_outside_the_planned_paths_is_refused(repo, tmp_path):
    model = _model(_edits(("calc.py", BASE, RIGHT), ("secrets.py", "", "x = 1\n"),
                          ("../escape.py", "", "x = 1\n"), ('calc.py"}, {"path', "", "x"),
                          ("tests/test_calc.py", TEST, "def test_x():\n    pass\n")))
    report = _run(repo, tmp_path, model)

    refused = [e["path"] for e in _audit(report) if e["step"] == "write_refused"]
    assert sorted(refused) == ["../escape.py", 'calc.py"}, {"path', "secrets.py", "tests/test_calc.py"]
    assert "secrets.py" not in report["diff"] and not (tmp_path / "escape.py").exists()
    assert report["status"] == "green"  # the planned write still went through


def test_stops_at_three_rounds_with_an_honest_failure_report(repo, tmp_path):
    report = _run(repo, tmp_path, _model(ADD_WRONG, _edits()))

    assert report["status"] == "failed" and report["rounds"] == 3
    assert [e["passed"] for e in _audit(report) if e["step"] == "tests"] == [False] * 4
    assert "did not reach green" in report["summary"] and "assert" in report["summary"]


def test_no_commit_push_or_merge_happens_in_any_path(repo, tmp_path):
    before = (_git(repo, "rev-list", "--all", "--count"), _git(repo, "branch", "--list"),
              _git(repo, "rev-parse", "HEAD"))
    _run(repo, tmp_path, _model(ADD_RIGHT))
    _run(repo, tmp_path, _model(ADD_WRONG, _edits()))
    after = (_git(repo, "rev-list", "--all", "--count"), _git(repo, "branch", "--list"),
             _git(repo, "rev-parse", "HEAD"))
    assert before == after
    assert _git(repo, "worktree", "list").count("\n") == 0  # worktrees cleaned up

    for forbidden in (["commit", "-m", "x"], ["push"], ["merge", "main"], ["-c", "a=b", "commit"]):
        with pytest.raises(agent.DevAgentError):
            agent.git(repo, *forbidden)


def test_the_entry_point_takes_ticket_text_or_a_spec_file(tmp_path):
    spec = tmp_path / "ticket.md"
    spec.write_text("# Add subtract\nIt returns a - b.\n", encoding="utf-8")
    assert agent.read_ticket(["--spec", str(spec)]) == spec.read_text(encoding="utf-8")
    assert agent.read_ticket(["Add", "subtract"]) == "Add subtract"


# ------------------------------------------------- edits, guards, feedback (#26 run)

def test_a_patch_is_edits_so_untouched_lines_are_never_retyped(repo, tmp_path):
    model = _model(ADD_RIGHT)
    report = _run(repo, tmp_path, model)
    assert report["status"] == "green"
    assert "Return search/replace edits" in model.prompts[-1]
    assert all(o.get("think") is False and o.get("max_tokens") == agent.MAX_TOKENS
               for o in model.options)
    # The line that already existed is untouched in the diff.
    assert "-    return a + b" not in report["diff"] and "+    return a - b" in report["diff"]


def test_an_edit_that_does_not_match_once_is_not_applied_and_is_fed_back(repo, tmp_path):
    missing = _edits(("calc.py", "def add(a,b):\n", "x"))  # not in the file as written
    model = _model(missing, ADD_RIGHT)
    report = _run(repo, tmp_path, model)

    assert report["status"] == "green" and report["rounds"] == 2
    [first] = [e for e in _audit(report) if e["step"] == "edit" and e["during"] == "patch_1"]
    assert first["paths"] == [] and "occurs 0 time(s)" in first["problems"][0]
    assert "were not applied" in model.prompts[-1] and "def add(a,b)" in model.prompts[-1]


def test_the_model_sees_its_own_changes_so_far(repo, tmp_path):
    model = _model(ADD_WRONG, FIX_WRONG)
    _run(repo, tmp_path, model)
    patch_prompts = [p for p in model.prompts if "Patch round" in p]
    assert "(none yet)" in patch_prompts[0]
    assert "+def subtract(a, b):" in patch_prompts[1] and "Undo any change" in patch_prompts[1]


def test_breaking_an_unrelated_line_is_caught_by_the_existing_tests(repo, tmp_path):
    """The failure the local model kept making: the acceptance test passes,
    but an untouched function is corrupted. The guard keeps it red."""
    corrupt = _edits(("calc.py", BASE, RIGHT.replace("return a + b", "return a * b")))
    model = _model(corrupt, _edits(("calc.py", "return a * b", "return a + b")))
    report = _run(repo, tmp_path, model)

    guards = [e["tests"] for e in _audit(report) if e["step"] == "guards"]
    assert guards == [["tests/test_add.py"]]
    runs = [(e["round"], e["passed"]) for e in _audit(report) if e["step"] == "tests"]
    assert runs == [(0, False), (1, False), (2, True)]  # round 1 was red only because of test_add
    patch_prompts = [p for p in model.prompts if "Patch round" in p]
    assert "tests/test_add.py" in patch_prompts[1] and "test_add" in patch_prompts[1]
    assert report["status"] == "green" and "1 existing test file" in report["summary"]


def test_the_plan_dedupes_its_test_file_and_refuses_to_reuse_an_existing_one(repo, tmp_path):
    listed_twice = {**PLAN, "files": ["calc.py", "tests/test_calc.py"]}
    report = _run(repo, tmp_path, _model(ADD_RIGHT, plan=listed_twice))
    assert report["status"] == "green"
    [plan] = [e["plan"] for e in _audit(report) if e["step"] == "plan"]
    assert plan["files"] == ["calc.py"]

    reuse = {**PLAN, "test_file": "tests/test_add.py"}
    report = _run(repo, tmp_path, _model(ADD_RIGHT, plan=reuse))
    assert report["status"] == "failed" and "already exists" in report["summary"]
    assert "test_add" in (repo / "tests" / "test_add.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("path", ['calc.py", "x', "calc.py}", "my file.py", "../calc.py", "C:/x.py", ""])
def test_garbled_or_escaping_paths_are_refused(path):
    assert agent._safe_path(path) is None
    assert agent._safe_path("services/research-writing/writing/x.py") == "services/research-writing/writing/x.py"


def test_the_test_writer_is_shown_how_tests_here_import_code(repo, tmp_path):
    model = _model(ADD_RIGHT)
    _run(repo, tmp_path, model)
    test_prompt = model.prompts[1]
    assert "tests/test_add.py (its imports and setup)" in test_prompt
    assert "sys.path.insert" in test_prompt and "def test_add" not in test_prompt


def test_crlf_files_keep_their_line_endings(repo, tmp_path):
    _git(repo, "config", "core.autocrlf", "false")  # keep the CRLF in the repo and the worktree
    (repo / "calc.py").write_bytes(BASE.replace("\n", "\r\n").encode())
    _git(repo, "-c", "user.email=t@example.org", "-c", "user.name=t", "commit", "-qam", "crlf")
    report = _run(repo, tmp_path, _model(ADD_RIGHT))
    assert report["status"] == "green"
    assert "-def add" not in report["diff"] and "+def subtract" in report["diff"]

```

## The experiment

```bash
cd Course-Project/code
pip install -e .
pytest -v tests/dev_agent/test_dev_agent.py
```

## Real result
Raw console output also saved as [`sample_run.txt`](./sample_run.txt) in this folder.

The Dev Agent graph's plan, test, patch structure, and its refusal rules, never overwrite an existing test, require red before green, pass against the real graph implementation above. Every live research run recorded in `Course-Project/docs/evidence/live-pass/` passed fully through the real 9 node writing graph, every run, not a subset.

## Why this is the real evidence, not a toy
The Dev Agent graph above is the same one used for real, on a real product ticket, in Lab 11.
