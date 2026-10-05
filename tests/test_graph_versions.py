"""Registered graph versions: revise under the same id, packages pin versions.

A registered graph starts at graph_version 1. ``PUT /api/graphs/{id}``
appends a new version guarded by ``expected_version`` (409 when stale, 404
when the id is unknown); reads and by-id previews follow the latest version,
while every sealed package stays bound to the version sealed against:
release re-reviews the bound history under the current registry and purges
judge dependency against that history, ignoring sensitivity marks. The
version history and the package bindings persist through a database reopen,
and an older database migrates its single snapshots to version 1.
"""

import json
import sqlite3
import threading

from fastapi.testclient import TestClient

from app import registry
from app.main import app
from app.storage import get_store, reset_store

client = TestClient(app)
# A storage fault must surface as a 500/INTERNAL response, not a test crash.
no_raise = TestClient(app, raise_server_exceptions=False)

# Version 1: the export copies users.email into mail.
GRAPH_V1 = {
    "nodes": [
        {
            "id": "users",
            "type": "source",
            "columns": [
                {"name": "uid", "sensitive": False},
                {"name": "email", "sensitive": False},
                {"name": "country", "sensitive": False},
            ],
        },
        {"id": "only_cn", "type": "filter", "input": "users",
         "references": ["country"]},
        {
            "id": "export_users",
            "type": "projection",
            "input": "only_cn",
            "columns": [
                {"name": "user_id", "copy": "uid"},
                {"name": "mail", "copy": "email"},
                {"name": "bucket", "constant": "ALL"},
            ],
        },
    ],
    "export": "export_users",
}

# Version 2: the mail column (and the email dependency) is dropped entirely.
GRAPH_V2 = {
    "nodes": [
        {
            "id": "users",
            "type": "source",
            "columns": [
                {"name": "uid", "sensitive": False},
                {"name": "email", "sensitive": False},
                {"name": "country", "sensitive": False},
            ],
        },
        {"id": "only_cn", "type": "filter", "input": "users",
         "references": ["country"]},
        {
            "id": "export_users",
            "type": "projection",
            "input": "only_cn",
            "columns": [
                {"name": "user_id", "copy": "uid"},
                {"name": "bucket", "constant": "ALL"},
            ],
        },
    ],
    "export": "export_users",
}

# Version 2 carrying an in-graph sensitive mark: blocked, but still valid.
GRAPH_V2_BLOCKED = {
    "nodes": [
        {
            "id": "users",
            "type": "source",
            "columns": [
                {"name": "uid", "sensitive": True},
                {"name": "email", "sensitive": False},
                {"name": "country", "sensitive": False},
            ],
        },
        {
            "id": "export_users",
            "type": "projection",
            "input": "users",
            "columns": [{"name": "user_id", "copy": "uid"}],
        },
    ],
    "export": "export_users",
}

ROWS_V1 = {
    "users": [
        {"uid": 1, "email": "a@x.example", "country": "CN"},
        {"uid": 2, "email": "b@x.example", "country": "US"},
    ]
}
ROWS_V2 = {
    "users": [
        {"uid": 1, "email": "a@x.example", "country": "CN"},
        {"uid": 2, "email": "b@x.example", "country": "US"},
    ]
}
CONDITIONS = {"only_cn": {"country": "CN"}}
ENTRIES_EMAIL = [{"source": "users", "column": "email"}]
ENTRIES_UID = [{"source": "users", "column": "uid"}]


def register(payload=GRAPH_V1):
    resp = client.post("/api/graphs", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def revise(gid, payload, expected_version, http=None):
    body = dict(payload)
    body["expected_version"] = expected_version
    return (http or client).put(f"/api/graphs/{gid}", json=body)


def revise_ok(gid, payload, expected_version):
    resp = revise(gid, payload, expected_version)
    assert resp.status_code == 200, resp.text
    return resp.json()


def put_registry(entries, generation):
    resp = client.put(
        "/api/sensitive-columns",
        json={"entries": entries, "expected_generation": generation},
    )
    assert resp.status_code == 200, resp.text
    return resp


def seal_ok(gid, rows=ROWS_V1, conditions=CONDITIONS):
    resp = client.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": rows, "conditions": conditions},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def release(pid, generation):
    return client.post(
        f"/api/packages/{pid}/release",
        json={"expected_generation": generation},
    )


# ----------------------------- version basics -------------------------------


def test_new_graph_and_migrated_graph_start_at_version_one():
    gid = register()
    got = client.get(f"/api/graphs/{gid}")
    assert got.status_code == 200
    assert got.json()["graph_version"] == 1


def test_revision_advances_version_under_the_same_id():
    gid = register()
    body = revise_ok(gid, GRAPH_V2, 1)
    assert set(body) == {"id", "graph_version", "generation", "review"}
    assert body["id"] == gid
    assert body["graph_version"] == 2
    assert body["generation"] == 0
    assert body["review"]["blocked"] is False

    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 2
    assert got["nodes"] == GRAPH_V2["nodes"]
    assert got["export"] == "export_users"
    # The old version content is retained alongside the new head.
    old = get_store().get_graph_version_row(gid, 1)
    assert json.loads(old["nodes"]) == GRAPH_V1["nodes"]
    head = get_store().get_graph_version_row(gid, 2)
    assert json.loads(head["nodes"]) == GRAPH_V2["nodes"]


def test_revision_saves_a_blocked_graph():
    gid = register()
    body = revise_ok(gid, GRAPH_V2_BLOCKED, 1)
    assert body["graph_version"] == 2
    assert body["review"]["blocked"] is True

    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 2
    assert got["nodes"] == GRAPH_V2_BLOCKED["nodes"]
    assert got["review"]["blocked"] is True


def test_stale_expected_version_returns_409_and_changes_nothing():
    gid = register()
    revise_ok(gid, GRAPH_V2, 1)

    resp = revise(gid, GRAPH_V2_BLOCKED, 1)
    assert resp.status_code == 409
    error = resp.json()["error"]
    assert error["code"] == "VERSION_CONFLICT"
    assert error["expected_version"] == 1
    assert error["current_version"] == 2

    # Nothing moved: still version 2, still the safe v2 content.
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 2
    assert got["nodes"] == GRAPH_V2["nodes"]
    # expected_version 0 is equally stale for a graph already at version 1.
    other = register()
    assert revise(other, GRAPH_V2, 0).status_code == 409
    assert client.get(f"/api/graphs/{other}").json()["graph_version"] == 1


def test_revision_unknown_graph_returns_404():
    resp = revise("does-not-exist", GRAPH_V2, 1)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "GRAPH_NOT_FOUND"
    # A stale version on a missing id is still a 404, never a 409.
    resp = client.put(
        "/api/graphs/does-not-exist",
        json={**GRAPH_V2, "expected_version": 99},
    )
    assert resp.status_code == 404


def test_revision_invalid_graph_is_rejected_without_storage_change():
    gid = register()
    bad = dict(GRAPH_V2)
    bad["nodes"] = [
        {"id": "p", "type": "projection", "input": "ghost",
         "columns": [{"name": "a", "copy": "a"}]}
    ]
    resp = revise(gid, bad, 1)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_REFERENCE"
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 1
    assert got["nodes"] == GRAPH_V1["nodes"]
    # No dangling version-2 history row either.
    assert get_store().get_graph_version_row(gid, 2) is None


def test_revision_schema_errors_are_422():
    gid = register()
    # Missing expected_version.
    resp = client.put(f"/api/graphs/{gid}", json=GRAPH_V2)
    assert resp.status_code == 422
    # Unknown field.
    resp = client.put(
        f"/api/graphs/{gid}", json={**GRAPH_V2, "expected_version": 1, "x": 1}
    )
    assert resp.status_code == 422
    # A negative expected_version violates the schema, never reaches storage.
    resp = revise(gid, GRAPH_V2, -1)
    assert resp.status_code == 422
    assert client.get(f"/api/graphs/{gid}").json()["graph_version"] == 1


def test_revision_never_advances_the_registry_generation():
    gid = register()
    put_registry(ENTRIES_UID, 0)  # generation 1; neither head depends on uid
    body = revise_ok(gid, GRAPH_V2, 1)
    assert body["generation"] == 1
    assert client.get("/api/sensitive-columns").json()["generation"] == 1
    assert client.get(f"/api/graphs/{gid}").json()["generation"] == 1


# ----------------------- reads and previews follow head ---------------------


def test_read_and_preview_use_the_latest_version():
    gid = register()
    seal_ok(gid)
    # v2 drops the email dependency; marking email only blocks v1.
    put_registry(ENTRIES_EMAIL, 0)
    revise_ok(gid, GRAPH_V2, 1)

    # The read follows the new head and is safe under the current registry.
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 2
    assert got["generation"] == 1
    assert got["review"]["blocked"] is False

    # By-id preview likewise runs the latest version and returns data.
    resp = client.post(
        f"/api/graphs/{gid}/preview",
        json={"rows": ROWS_V2, "conditions": CONDITIONS},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["columns"] == ["user_id", "bucket"]
    assert body["rows"] == [[1, "ALL"]]

    # A blocked head still gates the preview: revise to the in-graph marked v3.
    # The registry still only names users.email, so the mark is purely
    # in-graph and stays blocked regardless of the shared list.
    revise_ok(gid, GRAPH_V2_BLOCKED, 2)
    resp = client.post(
        f"/api/graphs/{gid}/preview",
        json={"rows": {"users": [
            {"uid": 1, "email": "a@x.example", "country": "CN"}]},
            "conditions": {}},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "PREVIEW_BLOCKED"


# ----------------------- packages bind the sealed version -------------------


def test_seal_binds_the_current_version_and_claim_reports_it():
    gid = register()
    pid = seal_ok(gid)
    sealed = client.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": ROWS_V1, "conditions": CONDITIONS},
    ).json()
    assert sealed["graph_version"] == 1

    revise_ok(gid, GRAPH_V2, 1)
    # A package sealed after the revision binds version 2.
    pid2 = seal_ok(gid, rows=ROWS_V2, conditions=CONDITIONS)
    assert client.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": ROWS_V2, "conditions": CONDITIONS},
    ).json()["graph_version"] == 2

    assert release(pid, 0).status_code == 200
    body = client.get(f"/api/packages/{pid}").json()
    assert body["graph_version"] == 1
    assert body["columns"] == ["user_id", "mail", "bucket"]
    assert body["rows"] == [[1, "a@x.example", "ALL"]]

    assert release(pid2, 0).status_code == 200
    body = client.get(f"/api/packages/{pid2}").json()
    assert body["graph_version"] == 2
    assert body["columns"] == ["user_id", "bucket"]


def test_revision_does_not_change_release_status_or_package_data():
    gid = register()
    pid = seal_ok(gid)
    assert release(pid, 0).status_code == 200
    assert client.get(f"/api/packages/{pid}").status_code == 200

    # A revision at the same generation leaves the release gate and the
    # frozen data exactly as they were.
    revise_ok(gid, GRAPH_V2, 1)
    resp = client.get(f"/api/packages/{pid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["graph_version"] == 1
    assert body["release_generation"] == 0
    assert body["rows"] == [[1, "a@x.example", "ALL"]]


def test_release_re_reviews_the_bound_history_not_the_current_head():
    gid = register()
    old_pkg = seal_ok(gid)  # bound to v1, which copies users.email

    # Mark email: v1 is blocked. Seal/release of the old package now fail.
    put_registry(ENTRIES_EMAIL, 0)
    assert release(old_pkg, 1).status_code == 400
    blocked = release(old_pkg, 1).json()["error"]
    assert blocked["code"] == "RELEASE_BLOCKED"
    mail = next(c for c in blocked["review"]["columns"] if c["name"] == "mail")
    assert mail["content_reasons"] == [{"node_id": "users", "column": "email"}]

    # Revise away the dependency; the new head is safe and seals/releases.
    revise_ok(gid, GRAPH_V2, 1)
    new_pkg = seal_ok(gid, rows=ROWS_V2, conditions=CONDITIONS)
    assert release(new_pkg, 1).status_code == 200
    assert client.get(f"/api/packages/{new_pkg}").json()["graph_version"] == 2

    # The old package's bound v1 stays blocked even though the head is safe.
    resp = release(old_pkg, 1)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "RELEASE_BLOCKED"
    # The old release was never recorded.
    assert client.get(f"/api/packages/{old_pkg}").status_code == 409


def test_purge_uses_the_bound_history_and_the_old_source_column():
    gid = register()
    old_pkg = seal_ok(gid)   # v1 export depends on users.email
    put_registry(ENTRIES_EMAIL, 0)
    revise_ok(gid, GRAPH_V2, 1)
    new_pkg = seal_ok(gid, rows=ROWS_V2, conditions=CONDITIONS)  # v2: no email

    # Purging by the old source column clears only the package bound to v1.
    resp = client.post(
        "/api/packages/purge",
        json={"columns": [{"source": "users", "column": "email"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["purged"] == [old_pkg]

    # The old package is gone for good (410); the v2 package is untouched and
    # still claimable after re-releasing under the current generation.
    assert client.get(f"/api/packages/{old_pkg}").status_code == 410
    assert release(new_pkg, 1).status_code == 200
    body = client.get(f"/api/packages/{new_pkg}").json()
    assert body["graph_version"] == 2
    assert body["rows"] == [[1, "ALL"]]

    # Clearing the registry (generation 2) cannot revive the purged package.
    put_registry([], 1)
    assert release(old_pkg, 2).status_code == 410
    assert client.get(f"/api/packages/{old_pkg}").status_code == 410


def test_seal_after_revision_preview_rules_apply_to_the_new_version():
    gid = register()
    put_registry(ENTRIES_EMAIL, 0)
    # The blocked v1 head cannot seal.
    resp = client.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": ROWS_V1, "conditions": CONDITIONS},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "PREVIEW_BLOCKED"
    revise_ok(gid, GRAPH_V2, 1)
    # Same registry, but the new head no longer depends on email: seals.
    pid = seal_ok(gid, rows=ROWS_V2, conditions=CONDITIONS)
    assert client.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": ROWS_V2, "conditions": CONDITIONS},
    ).status_code == 201
    assert release(pid, 1).status_code == 200


# ------------------------------ persistence ---------------------------------


def test_versions_and_bindings_survive_store_reopen(tmp_path):
    db_path = str(tmp_path / "versions.db")
    reset_store(db_path)
    gid = register()
    v1_pkg = seal_ok(gid)
    revise_ok(gid, GRAPH_V2, 1)
    v2_pkg = seal_ok(gid, rows=ROWS_V2, conditions=CONDITIONS)

    reset_store(db_path)
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 2
    assert got["nodes"] == GRAPH_V2["nodes"]
    store = get_store()
    assert json.loads(
        store.get_graph_version_row(gid, 1)["nodes"]
    ) == GRAPH_V1["nodes"]
    assert json.loads(
        store.get_graph_version_row(gid, 2)["nodes"]
    ) == GRAPH_V2["nodes"]
    assert store.get_package_row(v1_pkg)["graph_version"] == 1
    assert store.get_package_row(v2_pkg)["graph_version"] == 2


def test_old_database_migrates_its_single_snapshot_to_version_one(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    # Hand-build a database with the pre-versioning schema (no current_version
    # on graphs, no graph_versions table, no graph_version/purged on packages).
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta VALUES ('generation', '0');
        CREATE TABLE sensitive_entries (
            source_id TEXT NOT NULL, column_name TEXT NOT NULL,
            ordinal INTEGER NOT NULL, PRIMARY KEY (source_id, column_name));
        CREATE TABLE graphs (
            id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, nodes TEXT NOT NULL,
            export TEXT NOT NULL, review TEXT NOT NULL,
            review_generation INTEGER NOT NULL);
        CREATE TABLE packages (
            id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, graph_id TEXT NOT NULL,
            columns TEXT NOT NULL, rows TEXT NOT NULL, generation INTEGER NOT NULL,
            release_generation INTEGER);
        CREATE TABLE audit_log (
            seq INTEGER PRIMARY KEY, type TEXT NOT NULL, payload TEXT NOT NULL);
        """
    )
    review = {
        "export": "export_users", "blocked": False,
        "columns": [
            {"name": "user_id", "blocked": False,
             "content_reasons": [], "row_reasons": []},
            {"name": "mail", "blocked": False,
             "content_reasons": [], "row_reasons": []},
            {"name": "bucket", "blocked": False,
             "content_reasons": [], "row_reasons": []},
        ],
    }
    conn.execute(
        "INSERT INTO graphs VALUES (?, 0, ?, ?, ?, 0)",
        ("g1", json.dumps(GRAPH_V1["nodes"]), "export_users",
         json.dumps(review)),
    )
    conn.execute(
        "INSERT INTO packages VALUES (?, 0, 'g1', ?, ?, 0, NULL)",
        ("p1", json.dumps(["user_id", "mail", "bucket"]),
         json.dumps([[1, "a@x.example", "ALL"]])),
    )
    conn.commit()
    conn.close()

    # Reopening migrates: the legacy graph and package belong to version 1.
    reset_store(db_path)
    got = client.get("/api/graphs/g1").json()
    assert got["graph_version"] == 1
    assert got["nodes"] == GRAPH_V1["nodes"]
    assert got["review"]["blocked"] is False
    assert get_store().get_package_row("p1")["graph_version"] == 1

    # The migrated package behaves exactly like a version-1 seal: release and
    # claim serve the frozen rows, and marking the legacy dependency blocks it.
    assert release("p1", 0).status_code == 200
    body = client.get("/api/packages/p1").json()
    assert body["graph_version"] == 1
    assert body["rows"] == [[1, "a@x.example", "ALL"]]

    put_registry(ENTRIES_EMAIL, 0)
    assert release("p1", 1).status_code == 400
    assert client.post(
        "/api/packages/purge",
        json={"columns": [{"source": "users", "column": "email"}]},
    ).json()["purged"] == ["p1"]
    assert client.get("/api/packages/p1").status_code == 410


# ------------------------- failure leaves no trace --------------------------


class FaultyConnection:
    """Delegates to the real connection, failing an armed statement once."""

    def __init__(self, real):
        self._real = real
        self._predicate = None

    def arm(self, predicate):
        self._predicate = predicate

    def _check(self, sql):
        if self._predicate is not None and self._predicate(sql):
            self._predicate = None
            raise sqlite3.OperationalError("injected storage fault")

    def execute(self, sql, parameters=()):
        self._check(sql)
        return self._real.execute(sql, parameters)

    def executemany(self, sql, seq):
        self._check(sql)
        return self._real.executemany(sql, seq)

    def commit(self):
        self._check("COMMIT")
        return self._real.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _tb):
        if exc_type is None:
            self._real.commit()
        else:
            self._real.rollback()
        return False

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_failed_revision_write_leaves_no_trace():
    store = get_store()
    real = store._conn
    proxy = FaultyConnection(real)
    store._conn = proxy
    try:
        gid = register()
        proxy.arm(lambda sql: sql.startswith("INSERT INTO graph_versions"))
        resp = revise(gid, GRAPH_V2, 1, http=no_raise)
        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "INTERNAL"

        # The head, its version and the history are all untouched, on disk too.
        got = client.get(f"/api/graphs/{gid}").json()
        assert got["graph_version"] == 1
        assert got["nodes"] == GRAPH_V1["nodes"]
        assert store.get_graph_version_row(gid, 2) is None

        # Fault cleared: the very same revision succeeds.
        body = revise_ok(gid, GRAPH_V2, 1)
        assert body["graph_version"] == 2
    finally:
        store._conn = real


def test_commit_failure_during_revision_rolls_back_everything():
    store = get_store()
    real = store._conn
    proxy = FaultyConnection(real)
    store._conn = proxy
    try:
        gid = register()
        proxy.arm(lambda sql: sql == "COMMIT")
        resp = revise(gid, GRAPH_V2, 1, http=no_raise)
        assert resp.status_code == 500
        assert client.get(f"/api/graphs/{gid}").json()["graph_version"] == 1
        assert store.get_graph_version_row(gid, 2) is None
    finally:
        store._conn = real


# ----------------------- revision/replacement ordering ----------------------


def test_revision_and_replacement_take_effect_in_one_order(monkeypatch):
    """A replacement queued behind a revision re-reviews the new head.

    Hold the store lock inside the revision's review so a registry
    replacement queues behind it: the in-flight revision commits version 2
    first, then the replacement (marking users.email) re-reviews the *new*
    head — which no longer depends on email — and stores it as safe.
    """
    gid = register()
    entered = threading.Event()
    proceed = threading.Event()
    import app.registry as reg_mod

    original_review = reg_mod.run_review

    def slow_review(request, extra_sensitive):
        entered.set()
        assert proceed.wait(timeout=5)
        return original_review(request, extra_sensitive)

    monkeypatch.setattr(reg_mod, "run_review", slow_review)

    outcomes = {}

    def do_revise():
        outcomes["revise"] = registry.revise_graph(
            gid, GRAPH_V2["nodes"], GRAPH_V2["export"], 1
        )

    worker = threading.Thread(target=do_revise)
    worker.start()
    assert entered.wait(timeout=5)
    pending = threading.Thread(
        target=lambda: outcomes.setdefault(
            "replace",
            registry.replace_registry([("users", "email")], 0),
        )
    )
    pending.start()
    assert "replace" not in outcomes
    proceed.set()
    worker.join(timeout=5)
    pending.join(timeout=5)

    assert outcomes["revise"]["graph_version"] == 2
    assert outcomes["replace"]["generation"] == 1
    # The replacement reviewed v2 (no email dependency), so the head is safe.
    assert outcomes["replace"]["reviews"][gid]["blocked"] is False
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 2
    assert got["generation"] == 1
    assert got["review"]["blocked"] is False
