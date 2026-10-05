"""Registered-graph and shared-registry operations on top of the engine.

Every operation here runs under the store's single lock, so registration,
revisions, reads, registered previews, package sealing/releasing/claiming/
purging and registry replacements take effect in one observable total order.
A registered graph carries a version history under one fixed id: a revision
submits a new version with an optimistic ``expected_version`` (rejected as
409 when stale, as 404 when the id is unknown) and moves the graph's head;
reads and by-id previews always use the latest version, while a sealed
package stays bound to the exact version it sealed against — a revision never
changes the package's data, its release record or the graph a release/purge
re-evaluates. Within the lock each decision uses the *same* generation for
the registry, the stored review conclusions and the review a preview is
gated on: a preview either precedes a replacement (old registry plus old
conclusion) or follows it (new registry plus freshly computed, newly stored
conclusion), never a mixture of the two. Likewise a package is sealed,
released, claimed or purged either fully before or fully after a replacement
or a revision — a release can never be recorded under a stale generation,
and a claim can never serve data whose release generation the registry has
already moved past.
"""

from __future__ import annotations

import json
import uuid

from .engine import ValidationError
from .engine import export_dependencies
from .engine import preview as run_preview
from .engine import review as run_review
from .models import PreviewRequest, ReviewRequest
from .storage import get_store


class GraphNotFound(Exception):
    """No registered graph carries the given id (HTTP 404)."""


class StaleVersion(Exception):
    """The revision's expected version is outdated (HTTP 409).

    A new version is accepted only when ``expected_version`` equals the
    graph's current version; a stale value means another revision already
    advanced the version, and the whole submission is rejected without
    storing anything.
    """

    def __init__(self, expected: int, current: int):
        super().__init__(
            f"expected graph_version {expected} does not match current "
            f"graph_version {current}"
        )
        self.expected = expected
        self.current = current


class PackageNotFound(Exception):
    """No sealed export package carries the given id (HTTP 404)."""


class PackagePurged(Exception):
    """The package's sealed sample was purged by source column (HTTP 410).

    Purging is terminal: the package keeps its id but its rows and release
    record are gone for good, so release and claim both answer 410 and no
    registry change or restart can bring the sample back.
    """

    def __init__(self, package_id: str):
        super().__init__(
            f"package '{package_id}' was purged; its sealed sample is "
            "permanently gone"
        )
        self.package_id = package_id


class PackageNotReleased(Exception):
    """The package cannot be claimed under the current generation (HTTP 409).

    Either it was never released, or its release generation no longer equals
    the current registry generation — the registry moved on (even back to a
    safe list), so the package must be released again before it can be
    claimed.
    """

    def __init__(
        self,
        package_id: str,
        release_generation: int | None,
        current: int,
    ):
        if release_generation is None:
            message = f"package '{package_id}' has not been released"
        else:
            message = (
                f"package '{package_id}' was released under generation "
                f"{release_generation}, which does not match the current "
                f"generation {current}; it must be released again"
            )
        super().__init__(message)
        self.package_id = package_id
        self.release_generation = release_generation
        self.current = current


class StaleGeneration(Exception):
    """The replacement's expected generation is outdated (HTTP 409)."""

    def __init__(self, expected: int, current: int):
        super().__init__(
            f"expected generation {expected} does not match current "
            f"generation {current}"
        )
        self.expected = expected
        self.current = current


def _registry_snapshot() -> tuple[frozenset[tuple[str, str]], int]:
    db = get_store()
    return frozenset(db.get_entries()), db.get_generation()


def register_graph(
    nodes: list[dict[str, object]], export: str
) -> dict[str, object]:
    """Validate a whole submitted graph, review it under the current registry.

    The graph must pass exactly the same whole-graph validation as
    ``/api/review`` (an invalid graph is rejected, nothing is registered). Its
    sensitivity is the OR of the in-graph ``sensitive`` flags and the current
    shared registry. The original submitted ``nodes`` are stored verbatim.
    """
    # Parse/validate before taking the lock so model-level errors behave as on
    # the other endpoints; the engine re-validates under the registry snapshot.
    ReviewRequest.model_validate({"nodes": nodes, "export": export})

    db = get_store()
    with db.lock:
        extra_sensitive, generation = _registry_snapshot()
        request = ReviewRequest.model_validate(
            {"nodes": nodes, "export": export}
        )
        review_result = run_review(request, extra_sensitive)
        graph_id = uuid.uuid4().hex
        db.insert_graph(graph_id, nodes, export, review_result, generation)
        return {
            "id": graph_id,
            "nodes": nodes,
            "export": export,
            "generation": generation,
            "review": review_result,
        }


def revise_graph(
    graph_id: str,
    nodes: list[dict[str, object]],
    export: str,
    expected_version: int,
) -> dict[str, object]:
    """Submit a new version of a registered graph under its existing id.

    The submission runs exactly the same whole-graph validation as
    :func:`register_graph`: an invalid graph is rejected with the original
    ``/api/review`` error codes and nothing is stored. A *blocked* graph is
    still a valid graph and is saved like any other version — blocking never
    rejects a revision. The graph is reviewed under the current registry at
    the moment the revision takes effect; revisions never advance the
    registry generation and never touch sealed packages (their bound
    versions, frozen data and release records).

    Optimistic concurrency: ``expected_version`` must equal the graph's
    current version; a stale value raises :class:`StaleVersion` (HTTP 409)
    before validation and before any write, and an unknown id raises
    :class:`GraphNotFound` (HTTP 404). On success the version advances by one
    and the new head carries the freshly computed review.
    """
    # Schema-level parsing happens before taking the lock, as on every other
    # endpoint; the engine re-validates under the registry snapshot.
    ReviewRequest.model_validate({"nodes": nodes, "export": export})

    db = get_store()
    with db.lock:
        row = db.get_graph_row(graph_id)
        if row is None:
            raise GraphNotFound(graph_id)
        current_version = row["current_version"]
        if expected_version != current_version:
            raise StaleVersion(expected_version, current_version)
        extra_sensitive, generation = _registry_snapshot()
        request = ReviewRequest.model_validate(
            {"nodes": nodes, "export": export}
        )
        review_result = run_review(request, extra_sensitive)
        next_version = current_version + 1
        # History append and head move commit together or not at all.
        db.revise_graph(
            graph_id, nodes, export, review_result, generation, next_version
        )
        return {
            "id": graph_id,
            "graph_version": next_version,
            "generation": generation,
            "review": review_result,
        }


def read_graph(graph_id: str) -> dict[str, object]:
    """Return the latest version plus the current generation and review."""
    db = get_store()
    with db.lock:
        row = db.get_graph_row(graph_id)
        if row is None:
            raise GraphNotFound(graph_id)
        return {
            "id": row["id"],
            "nodes": _loads(row["nodes"]),
            "export": row["export"],
            "graph_version": row["current_version"],
            "generation": row["review_generation"],
            "review": _loads(row["review"]),
        }


def preview_graph(
    graph_id: str,
    rows: dict[str, list[dict[str, object]]],
    conditions: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Preview a registered graph from sample inputs and the current registry.

    The stored graph supplies ``nodes``/``export``; the caller supplies only
    ``rows``/``conditions`` and all original preview rules apply unchanged.
    The graph is re-evaluated under the registry snapshot taken under the same
    lock as the stored review conclusions, so the block decision can never be
    based on a different registry than the conclusion returned.
    """
    db = get_store()
    with db.lock:
        row = db.get_graph_row(graph_id)
        if row is None:
            raise GraphNotFound(graph_id)
        extra_sensitive, generation = _registry_snapshot()
        request = PreviewRequest.model_validate(
            {
                "nodes": _loads(row["nodes"]),
                "export": row["export"],
                "rows": rows,
                "conditions": conditions,
            }
        )
        result = run_preview(request, extra_sensitive)
        result["id"] = graph_id
        result["generation"] = generation
        return result


# ---------------------------------------------------------------------------
# Sealed export packages: seal, release, claim
# ---------------------------------------------------------------------------


def seal_package(
    graph_id: str,
    rows: dict[str, list[dict[str, object]]],
    conditions: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Seal a registered graph's small-sample result into an export package.

    The same registry snapshot and exactly the same preview rules as
    :func:`preview_graph` apply: any input error (coverage, key sets,
    scalars, row limits) or a blocked export rejects the request *before*
    anything is stored, so a failed seal never leaves a package behind. Only
    when the export passes review are the materialized ``columns``/``rows``,
    the current registry generation and the graph's *current version*
    persisted; later revisions never rebind the package. The response carries
    just the new package id, the bound graph version and that creation
    generation.
"""
    db = get_store()
    with db.lock:
        row = db.get_graph_row(graph_id)
        if row is None:
            raise GraphNotFound(graph_id)
        graph_version = row["current_version"]
        extra_sensitive, generation = _registry_snapshot()
        request = PreviewRequest.model_validate(
            {
                "nodes": _loads(row["nodes"]),
                "export": row["export"],
                "rows": rows,
                "conditions": conditions,
            }
        )
        result = run_preview(request, extra_sensitive)
        package_id = uuid.uuid4().hex
        db.insert_package(
            package_id,
            graph_id,
            graph_version,
            result["columns"],
            result["rows"],
            generation,
        )
        return {
            "id": package_id,
            "graph_version": graph_version,
            "generation": generation,
        }


def release_package(
    package_id: str, expected_generation: int
) -> dict[str, object]:
    """Release a sealed package under the current registry generation.

    ``expected_generation`` must equal the current registry generation; a
    stale value is rejected as :class:`StaleGeneration` (HTTP 409) without
    recording anything. The graph *version the package was sealed from* is
    re-reviewed under the current registry snapshot (the graph may have been
    revised meanwhile — the current head never takes the bound version's
    place): when that historical version is blocked the release is rejected
    (HTTP 400) and nothing is recorded. On success the current generation is
    stored as the package's release generation and only the package id and
    that generation are returned. Re-releasing after a registry change
    overwrites the previous release generation. A purged package can never be
    released again (:class:`PackagePurged`, HTTP 410).
"""
    db = get_store()
    with db.lock:
        row = db.get_package_row(package_id)
        if row is None:
            raise PackageNotFound(package_id)
        if row["purged"]:
            raise PackagePurged(package_id)
        extra_sensitive, generation = _registry_snapshot()
        if expected_generation != generation:
            raise StaleGeneration(expected_generation, generation)
        graph_version = row["graph_version"]
        version_row = db.get_graph_version_row(row["graph_id"], graph_version)
        if version_row is None:  # pragma: no cover - packages bind kept history
            raise GraphNotFound(row["graph_id"])
        review_result = run_review(
            ReviewRequest.model_validate(
                {
                    "nodes": _loads(version_row["nodes"]),
                    "export": version_row["export"],
                }
            ),
            extra_sensitive,
        )
        if review_result["blocked"]:
            raise ValidationError(
                "RELEASE_BLOCKED",
                "the graph version bound to this package is blocked under "
                "the current registry; the package is not released",
                version_row["export"],
                {"review": review_result},
            )
        db.release_package(package_id, generation)
        return {"id": package_id, "release_generation": generation}


def claim_package(package_id: str) -> dict[str, object]:
    """Hand out a sealed package's frozen sample data.

    Nothing is recomputed: the stored ``columns``/``rows`` are returned
    together with the graph id, the graph version the package was sealed
    from, the creation generation and the release generation. A package that
    was never released, or whose release generation no longer equals the
    current registry generation (the registry moved on — even back to a safe
    list), is rejected as :class:`PackageNotReleased` (HTTP 409) without any
    sample data. A purged package is rejected as :class:`PackagePurged`
    (HTTP 410) forever.
"""
    db = get_store()
    with db.lock:
        row = db.get_package_row(package_id)
        if row is None:
            raise PackageNotFound(package_id)
        if row["purged"]:
            raise PackagePurged(package_id)
        current = db.get_generation()
        release_generation = row["release_generation"]
        if release_generation is None or release_generation != current:
            raise PackageNotReleased(package_id, release_generation, current)
        return {
            "id": row["id"],
            "graph_id": row["graph_id"],
            "graph_version": row["graph_version"],
            "generation": row["generation"],
            "release_generation": release_generation,
            "columns": _loads(row["columns"]),
            "rows": _loads(row["rows"]),
        }


# ---------------------------------------------------------------------------
# Purging sealed samples by source column
# ---------------------------------------------------------------------------


def _source_columns(pairs: set[tuple[str, str]] | frozenset[tuple[str, str]]):
    return [
        {"source": source_id, "column": column_name}
        for source_id, column_name in sorted(pairs)
    ]


def purge_packages(columns: list[tuple[str, str]]) -> dict[str, object]:
    """Purge every live package whose export depends on a listed column.

    The dependency decision follows the original content-provenance and
    row-influence rules with sensitivity marks ignored entirely (neither
    in-graph ``sensitive`` flags nor the shared registry play a role), and it
    runs against the graph *version the package was sealed from* — the
    current head never takes the bound history's place: a package is purged
    when any listed ``(source, column)`` pair reaches its bound version's
    export — through copied content, or as row-level influence left by a
    filter reference or a join key, which a constant replacement cannot cut.
    Only the export's dependency closure counts; side branches never matter.

    Only packages that already exist when this operation takes effect are
    considered; packages sealed afterwards are unaffected. Packages already
    purged are never listed again, and a purge that matches nothing changes
    nothing — and, because it changes nothing, leaves no audit record. The
    graph, the shared registry and the generation are never touched. A
    matching purge commits its package updates together with one audit row
    (current generation, requested source columns sorted by
    ``(source, column)`` and the actually purged package ids sorted) under
    the same lock as seal/release/claim, so a failed write or commit leaves
    every package and the log exactly as they were.
    """
    db = get_store()
    targets = frozenset(columns)
    with db.lock:
        # Snapshot the still-live packages grouped by graph AND the bound
        # version: a revision never rebinds a package, so the dependency
        # decision always runs against the historical graph that was sealed.
        # Already purged packages are never cleared (or listed) a second time.
        live_by_version: dict[tuple[str, int], list[str]] = {}
        for row in db.list_package_rows():
            if not row["purged"]:
                live_by_version.setdefault(
                    (row["graph_id"], row["graph_version"]), []
                ).append(row["id"])

        purged: list[str] = []
        for (graph_id, graph_version), package_ids in live_by_version.items():
            version_row = db.get_graph_version_row(graph_id, graph_version)
            if version_row is None:  # pragma: no cover - packages bind kept history
                continue
            request = ReviewRequest.model_validate(
                {
                    "nodes": _loads(version_row["nodes"]),
                    "export": version_row["export"],
                }
            )
            if export_dependencies(request) & targets:
                purged.extend(package_ids)

        purged.sort()
        if purged:
            audit_payload = {
                "generation": db.get_generation(),
                "columns": _source_columns(targets),
                "package_ids": purged,
            }
            db.purge_packages(purged, audit_payload)
        return {"purged": purged}


def get_registry() -> dict[str, object]:
    db = get_store()
    with db.lock:
        return {
            "generation": db.get_generation(),
            "entries": [
                {"source": source_id, "column": column_name}
                for source_id, column_name in db.get_entries()
            ],
        }


def replace_registry(
    entries: list[tuple[str, str]], expected_generation: int
) -> dict[str, object]:
    """Replace the whole registry and re-review every registered graph.

    Stale ``expected_generation`` is rejected as :class:`StaleGeneration`
    without touching anything and without leaving an audit record. Otherwise
    every registered graph is re-reviewed under the OR of its in-graph flags
    and the new entries, and entries, the generation increment, all new
    conclusions and one audit-log row are committed in one transaction; any
    failure rolls the whole replacement back. The audit row records the old
    and new generation plus the source columns the replacement adds and
    removes (each sorted by ``(source, column)``) — a replacement whose
    contents are identical to the current list still changes the generation
    and is still recorded, with empty added/removed lists.
    """
    db = get_store()
    with db.lock:
        current = db.get_generation()
        if expected_generation != current:
            raise StaleGeneration(expected_generation, current)

        old_entries = frozenset(db.get_entries())
        new_entries = frozenset(entries)
        extra_sensitive = new_entries
        new_reviews: dict[str, dict[str, object]] = {}
        for row in db.list_graph_rows():
            request = ReviewRequest.model_validate(
                {"nodes": json.loads(row["nodes"]), "export": row["export"]}
            )
            new_reviews[row["id"]] = run_review(request, extra_sensitive)

        next_generation = current + 1
        audit_payload = {
            "generation_before": current,
            "generation_after": next_generation,
            "added": _source_columns(new_entries - old_entries),
            "removed": _source_columns(old_entries - new_entries),
        }
        # Entries, generation, conclusions and the audit row commit together
        # or not at all.
        db.replace_entries(entries, next_generation, new_reviews, audit_payload)
        return {
            "generation": next_generation,
            "entries": [
                {"source": source_id, "column": column_name}
                for source_id, column_name in entries
            ],
            "reviews": {graph_id: review for graph_id, review in new_reviews.items()},
        }


class UntilSeqOutOfRange(Exception):
    """An explicit ``until_seq`` exceeds the largest committed seq (HTTP 400)."""

    def __init__(self, until_seq: int, max_seq: int):
        super().__init__(
            f"until_seq {until_seq} is larger than the latest committed "
            f"seq {max_seq}"
        )
        self.until_seq = until_seq
        self.max_seq = max_seq


def read_audit_log(
    after_seq: int, limit: int, until_seq: int | None = None
) -> dict[str, object]:
    """Return one page of the committed audit log, seq ascending.

    The page covers ``after_seq < seq <= until_seq``; when ``until_seq`` is
    omitted the cutoff defaults to the largest committed seq (0 on an empty
    log), and an explicit cutoff above that value is rejected as
    :class:`UntilSeqOutOfRange` so a fixed cutoff can never silently include
    operations committed later. The cutoff is echoed back alongside the
    records and the next cursor — the last returned seq, or ``after_seq``
    itself when the page is empty.
    """
    db = get_store()
    with db.lock:
        max_seq = db.max_audit_seq()
        if until_seq is None:
            cutoff = max_seq
        elif until_seq > max_seq:
            raise UntilSeqOutOfRange(until_seq, max_seq)
        else:
            cutoff = until_seq
        records = db.list_audit_rows(after_seq, cutoff, limit)
        next_cursor = records[-1]["seq"] if records else after_seq
        return {
            "until_seq": cutoff,
            "records": records,
            "next_cursor": next_cursor,
        }


def _loads(value: str) -> object:
    return json.loads(value)
