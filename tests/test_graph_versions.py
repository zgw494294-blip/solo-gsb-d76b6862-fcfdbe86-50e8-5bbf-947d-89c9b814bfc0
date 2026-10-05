"""Registered graph versions: PUT /api/graphs/{id}.

A registered graph accepts new versions under its original id; packages keep
binding the version that was latest when they were sealed. Reads and by-id
previews use the latest version; release and purge reason over the bound
historical version combined with the current registry.
"""

import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from app import registry
from app.main import app
from app.storage import get_store, reset_store

client = TestClient(app)
# A storage fault must surface as a 500/INTERNAL response, not a test crash.
no_raise = TestClient(app, raise_server_exceptions=False)

# V1: the usual users -> filter(country) -> projection export; copying email.
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
ROWS_V1 = {
    "users": [
        {"uid": 1, "email": "a@x.example", "country": "CN"},
        {"uid": 2, "email": "b@x.example", "country": "US"},
    ]
}
CONDITIONS_V1 = {"only_cn": {"country": "CN"}}

# V2: the export no longer copies email (it is replaced by a constant), so a
# registry entry on users/email no longer blocks this version.
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
                {"name": "mail", "constant": None},
                {"name": "bucket", "constant": "ALL"},
            ],
        },
    ],
    "export": "export_users",
}


def register(payload=GRAPH_V1):
    resp = client.post("/api/graphs", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def put_version(gid, version, payload, http=None):
    return (http or client).put(
        f"/api/graphs/{gid}",
        json={**payload, "expected_version": version},
    )


def replace_v2(gid, expected=1, http=None):
    return put_version(gid, expected, GRAPH_V2, http=http)


def seal(gid, rows=ROWS_V1, conditions=CONDITIONS_V1, http=None):
    return (http or client).post(
        f"/api/graphs/{gid}/packages",
        json={"rows": rows, "conditions": conditions},
    )


def seal_ok(gid, rows=ROWS_V1, conditions=CONDITIONS_V1):
    resp = seal(gid, rows, conditions)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def release(pid, generation, http=None):
    return (http or client).post(
        f"/api/packages/{pid}/release",
        json={"expected_generation": generation},
    )


def put_registry(entries, generation):
    resp = client.put(
        "/api/sensitive-columns",
        json={"entries": entries, "expected_generation": generation},
    )
    assert resp.status_code == 200, resp.text
    return resp


# --------------------------- new-version basics -----------------------------


def test_registration_starts_at_version_one():
    body = register()
    assert body["graph_version"] == 1
    assert set(body) == {
        "id", "graph_version", "nodes", "export", "generation", "review"}


def test_replace_commits_next_version_and_replaces_the_latest_graph():
    gid = register()["id"]
    resp = replace_v2(gid)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Success returns exactly id, graph_version, generation and review.
    assert set(body) == {"id", "graph_version", "generation", "review"}
    assert body["id"] == gid
    assert body["graph_version"] == 2
    assert body["generation"] == 0
    assert body["review"]["blocked"] is False

    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 2
    assert got["nodes"] == GRAPH_V2["nodes"]
    assert got["export"] == "export_users"
    assert got["generation"] == 0

    # Versions keep advancing by one under the same id.
    resp = replace_v2(gid, expected=2)
    assert resp.status_code == 200
    assert resp.json()["graph_version"] == 3
    assert client.get(f"/api/graphs/{gid}").json()["graph_version"] == 3


def test_stale_expected_version_returns_409_and_changes_nothing():
    gid = register()["id"]
    assert replace_v2(gid, expected=1).status_code == 200
    # Replaying the old expected version is rejected.
    resp = replace_v2(gid, expected=1)
    assert resp.status_code == 409
    error = resp.json()["error"]
    assert error["code"] == "VERSION_CONFLICT"
    assert error["expected_version"] == 1
    assert error["current_version"] == 2
    # The stored version stays at 2 with the v2 graph.
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 2
    assert got["nodes"] == GRAPH_V2["nodes"]
    # A future expectation is just as stale.
    resp = replace_v2(gid, expected=99)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "VERSION_CONFLICT"


def test_replace_unknown_graph_returns_404():
    resp = put_version("does-not-exist", 0, GRAPH_V2)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "GRAPH_NOT_FOUND"


def test_replace_invalid_graph_is_rejected_without_version_change():
    gid = register()["id"]
    bad = {"nodes": [{"id": "p", "type": "projection", "input": "ghost",
                      "columns": [{"name": "a", "copy": "a"}]}],
           "export": "p"}
    resp = put_version(gid, 1, bad)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_REFERENCE"
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 1
    assert got["nodes"] == GRAPH_V1["nodes"]


def test_replace_schema_errors_are_422_and_change_nothing():
    gid = register()["id"]
    # Missing expected_version.
    resp = client.put(f"/api/graphs/{gid}", json=GRAPH_V2)
    assert resp.status_code == 422
    # Negative expected_version.
    resp = put_version(gid, -1, GRAPH_V2)
    assert resp.status_code == 422
    # Unknown field.
    resp = client.put(
        f"/api/graphs/{gid}",
        json={**GRAPH_V2, "expected_version": 1, "nope": 1},
    )
    assert resp.status_code == 422
    assert client.get(f"/api/graphs/{gid}").json()["graph_version"] == 1


def test_blocked_new_version_is_still_saved():
    gid = register()["id"]
    # Mark users.email: v2 still copies email, so it is blocked under the
    # current registry — the version must nevertheless be saved.
    put_registry([{"source": "users", "column": "email"}], 0)
    blocked_v2 = {
        "nodes": [
            {
                "id": "users", "type": "source",
                "columns": [
                    {"name": "uid", "sensitive": False},
                    {"name": "email", "sensitive": False},
                    {"name": "country", "sensitive": False},
                ],
            },
            {"id": "only_cn", "type": "filter", "input": "users",
             "references": ["country"]},
            {
                "id": "export_users", "type": "projection",
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
    resp = put_version(gid, 1, blocked_v2)
    assert resp.status_code == 200
    body = resp.json()
    assert body["graph_version"] == 2
    assert body["generation"] == 1
    assert body["review"]["blocked"] is True
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 2
    assert got["generation"] == 1
    assert got["review"]["blocked"] is True


def test_revision_does_not_change_generation_or_audit_log():
    gid = register()["id"]
    put_registry([{"source": "users", "column": "email"}], 0)  # gen 1
    assert replace_v2(gid, expected=1).status_code == 200
    # The revision moved no generation and wrote no audit record.
    assert client.get("/api/sensitive-columns").json()["generation"] == 1
    assert client.get("/api/audit-log").json()["until_seq"] == 1


# ------------------------- reads/previews track latest ----------------------


def test_preview_response_keys_are_unchanged_from_the_legacy_shape():
    gid = register()["id"]
    replace_v2(gid, expected=1)
    resp = client.post(
        f"/api/graphs/{gid}/preview",
        json={"rows": ROWS_V1, "conditions": CONDITIONS_V1},
    )
    assert resp.status_code == 200
    # The by-id preview keeps its old response shape: graph_version rides on
    # read/seal/claim responses, not on preview responses.
    assert set(resp.json()) == {
        "export", "columns", "rows", "review", "id", "generation"}


def test_read_carries_graph_version_and_latest_submission():
    gid = register()["id"]
    got = client.get(f"/api/graphs/{gid}")
    assert got.status_code == 200
    body = got.json()
    assert set(body) == {
        "id", "graph_version", "nodes", "export", "generation", "review"}
    assert body["graph_version"] == 1
    replace_v2(gid, expected=1)
    assert client.get(f"/api/graphs/{gid}").json()["graph_version"] == 2


def test_preview_by_id_uses_the_latest_version():
    gid = register()["id"]
    # Seal and preview while v1 is latest.
    assert seal(gid).status_code == 201
    resp = client.post(
        f"/api/graphs/{gid}/preview",
        json={"rows": ROWS_V1, "conditions": CONDITIONS_V1},
    )
    assert resp.status_code == 200
    assert resp.json()["columns"] == ["user_id", "mail", "bucket"]

    # After the revision the preview addresses v2 (mail is now null).
    assert replace_v2(gid, expected=1).status_code == 200
    resp = client.post(
        f"/api/graphs/{gid}/preview",
        json={"rows": ROWS_V1, "conditions": CONDITIONS_V1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["columns"] == ["user_id", "mail", "bucket"]
    assert body["rows"] == [[1, None, "ALL"]]


# -------------------- packages bind the version at seal time ----------------


def test_seal_binds_and_reports_the_current_version():
    gid = register()["id"]
    resp = seal(gid)
    assert resp.status_code == 201
    assert resp.json()["graph_version"] == 1
    # A package sealed after the revision binds version 2.
    assert replace_v2(gid, expected=1).status_code == 200
    resp = seal(gid)
    assert resp.status_code == 201
    assert resp.json()["graph_version"] == 2


def test_release_reviews_the_bound_old_version_under_current_registry():
    gid = register()["id"]
    pid = seal_ok(gid)  # binds v1
    # Revise to the email-independent v2, then mark users.email at gen 1.
    assert replace_v2(gid, expected=1).status_code == 200
    put_registry([{"source": "users", "column": "email"}], 0)

    # The current graph (v2) is safe and seals/releases fine ...
    assert seal(gid).status_code == 201
    # ... but the v1 package is still blocked through its bound version.
    resp = release(pid, 1)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "RELEASE_BLOCKED"
    mail = next(c for c in error["review"]["columns"] if c["name"] == "mail")
    assert mail["content_reasons"] == [
        {"node_id": "users", "column": "email"}]
    assert client.get(f"/api/packages/{pid}").status_code == 409

    # Clear the registry (gen 2): the bound v1 review is safe again and the
    # very same package releases; claim carries the bound version.
    put_registry([], 1)
    released = release(pid, 2)
    assert released.status_code == 200
    # The release response keeps its old shape (no graph_version).
    assert released.json() == {"id": pid, "release_generation": 2}
    body = client.get(f"/api/packages/{pid}").json()
    assert body["graph_version"] == 1
    assert body["generation"] == 0
    assert body["release_generation"] == 2
    assert body["rows"] == [[1, "a@x.example", "ALL"]]


def test_new_version_package_releases_while_old_one_stays_blocked():
    gid = register()["id"]
    old_pid = seal_ok(gid)
    assert replace_v2(gid, expected=1).status_code == 200
    new_pid = seal_ok(gid)
    put_registry([{"source": "users", "column": "email"}], 0)

    # The v2 package releases under the current (email-marked) registry ...
    assert release(new_pid, 1).status_code == 200
    assert client.get(f"/api/packages/{new_pid}").status_code == 200
    assert client.get(f"/api/packages/{new_pid}").json()["graph_version"] == 2
    # ... while the v1 package remains blocked.
    assert release(old_pid, 1).status_code == 400


def test_revision_does_not_change_package_data_or_release_state():
    gid = register()["id"]
    pid = seal_ok(gid)
    release(pid, 0)
    assert client.get(f"/api/packages/{pid}").status_code == 200

    assert replace_v2(gid, expected=1).status_code == 200
    # Still released at generation 0; the frozen data is byte-identical.
    body = client.get(f"/api/packages/{pid}").json()
    assert body["release_generation"] == 0
    assert body["rows"] == [[1, "a@x.example", "ALL"]]
    assert body["columns"] == ["user_id", "mail", "bucket"]


def test_registry_replacement_re_reviews_only_latest_versions():
    gid = register()["id"]
    assert replace_v2(gid, expected=1).status_code == 200
    other = register(GRAPH_V1)["id"]
    resp = put_registry([{"source": "users", "column": "email"}], 0)
    reviews = resp.json()["reviews"]
    # The revised graph (v2, email constant-nulled) stays safe ...
    assert reviews[gid]["blocked"] is False
    # ... while the still-v1 other graph is blocked.
    assert reviews[other]["blocked"] is True
    assert client.get(f"/api/graphs/{gid}").json()["review"]["blocked"] is False


# -------------------- purge reasons over the bound version ------------------


def test_purge_judges_the_bound_historical_version():
    gid = register()["id"]
    v1_package = seal_ok(gid)  # v1 depends on users/email via copy
    assert replace_v2(gid, expected=1).status_code == 200
    v2_package = seal_ok(gid)  # v2 no longer copies email

    # users/email matches the v1 package's bound graph but not v2's.
    resp = client.post(
        "/api/packages/purge",
        json={"columns": [{"source": "users", "column": "email"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["purged"] == [v1_package]
    assert client.get(f"/api/packages/{v1_package}").status_code == 410
    assert client.get(f"/api/packages/{v2_package}").status_code == 409

    # users/country (filter reference -> row influence in v2 too) hits v2.
    resp = client.post(
        "/api/packages/purge",
        json={"columns": [{"source": "users", "column": "country"}]},
    )
    assert resp.json()["purged"] == [v2_package]
    assert client.get(f"/api/packages/{v2_package}").status_code == 410
    # The already-purged v1 package is not listed a second time.


def test_revision_cannot_spare_or_purge_an_already_sealed_package():
    gid = register()["id"]
    pid = seal_ok(gid)  # v1 copies email
    # Revise away the email dependency, then purge by email: the package is
    # still judged on its bound v1 graph and is purged.
    assert replace_v2(gid, expected=1).status_code == 200
    resp = client.post(
        "/api/packages/purge",
        json={"columns": [{"source": "users", "column": "email"}]},
    )
    assert resp.json()["purged"] == [pid]
    assert client.get(f"/api/packages/{pid}").status_code == 410


# ------------------------------- persistence ---------------------------------


def test_versions_and_bindings_survive_store_reopen(tmp_path):
    db_path = str(tmp_path / "versions.db")
    reset_store(db_path)
    gid = register()["id"]
    v1_pid = seal_ok(gid)
    release(v1_pid, 0)
    assert replace_v2(gid, expected=1).status_code == 200
    v2_pid = seal_ok(gid)

    # Simulate a restart: drop the process-wide store, reopen the same file.
    reset_store(db_path)
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 2
    assert got["nodes"] == GRAPH_V2["nodes"]

    body = client.get(f"/api/packages/{v1_pid}").json()
    assert body["graph_version"] == 1
    assert body["rows"] == [[1, "a@x.example", "ALL"]]
    assert client.get(f"/api/packages/{v2_pid}").status_code == 409  # unreleased
    binding = get_store().get_package_row(v2_pid)
    assert binding["graph_version"] == 2

    # The bound v1 history row is still readable after reopen.
    version_row = get_store().get_graph_version_row(gid, 1)
    assert version_row["export"] == "export_users"
    assert version_row["nodes"] != get_store().get_graph_row(gid)["nodes"]


def test_database_from_before_versioning_binds_old_graphs_to_version_one(
    tmp_path,
):
    db_path = str(tmp_path / "old.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta VALUES ('generation', '0');
        CREATE TABLE sensitive_entries (
            source_id TEXT NOT NULL, column_name TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (source_id, column_name));
        CREATE TABLE graphs (
            id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL,
            nodes TEXT NOT NULL, export TEXT NOT NULL,
            review TEXT NOT NULL, review_generation INTEGER NOT NULL);
        CREATE TABLE packages (
            id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL,
            graph_id TEXT NOT NULL, columns TEXT NOT NULL, rows TEXT NOT NULL,
            generation INTEGER NOT NULL, release_generation INTEGER);
        INSERT INTO graphs VALUES
            ('g1', 0, '[{"id":"s","type":"source",'
             || '"columns":[{"name":"a"},{"name":"b"}]}]',
             's', '{"blocked": false}', 0);
        INSERT INTO packages VALUES
            ('p1', 0, 'g1', '["a"]', '[[1]]', 0, 0);
        """
    )
    conn.close()

    reset_store(db_path)
    # The old graph is at version 1 with its snapshot backfilled into the
    # history table; the old package binds version 1.
    row = get_store().get_graph_row("g1")
    assert row["version"] == 1
    history = get_store().get_graph_version_row("g1", 1)
    assert history is not None and history["export"] == "s"
    package = get_store().get_package_row("p1")
    assert package["graph_version"] == 1
    assert package["purged"] == 0

    # It behaves like a current version-1 graph through the HTTP API.
    got = client.get("/api/graphs/g1").json()
    assert got["graph_version"] == 1
    assert client.get("/api/packages/p1").status_code == 200
    # And a new version can be committed on top of the migrated one.
    new_graph = {
        "nodes": [{"id": "s", "type": "source",
                   "columns": [{"name": "a"}]}],
        "export": "s",
    }
    resp = put_version("g1", 1, new_graph)
    assert resp.status_code == 200
    assert resp.json()["graph_version"] == 2


# --------------------- failed commit leaves no trace ------------------------


class FaultyConnection:
    """Delegates to the real connection, failing an armed statement once."""

    def __init__(self, real):
        self._real = real
        self._predicate = None

    def arm(self, predicate):
        self._predicate = predicate

    def _check(self, sql):
        if self._predicate is not None and self._predicate(sql):
            self._predicate = None  # a transient fault fires exactly once
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


@pytest.fixture
def faulty_conn():
    store = get_store()
    real = store._conn
    proxy = FaultyConnection(real)
    store._conn = proxy
    yield proxy
    store._conn = real


def test_failed_replace_update_leaves_the_old_version(faulty_conn):
    gid = register()["id"]
    # Fail the transaction's first UPDATE (the latest-version pointer).
    faulty_conn.arm(lambda sql: sql.startswith("UPDATE graphs"))
    resp = replace_v2(gid, expected=1, http=no_raise)
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "INTERNAL"
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 1
    assert got["nodes"] == GRAPH_V1["nodes"]
    # No half-written history row either.
    assert get_store().get_graph_version_row(gid, 2) is None

    # Fault cleared: the same expected version still works.
    resp = replace_v2(gid, expected=1)
    assert resp.status_code == 200
    assert resp.json()["graph_version"] == 2


def test_failed_replace_history_insert_rolls_the_pointer_back(faulty_conn):
    gid = register()["id"]
    faulty_conn.arm(lambda sql: sql.startswith("INSERT INTO graph_versions"))
    resp = replace_v2(gid, expected=1, http=no_raise)
    assert resp.status_code == 500
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 1
    assert got["nodes"] == GRAPH_V1["nodes"]
    assert get_store().get_graph_version_row(gid, 2) is None

    # Fault cleared: the same expected version still works.
    resp = replace_v2(gid, expected=1)
    assert resp.status_code == 200
    assert resp.json()["graph_version"] == 2


def test_failed_replace_commit_rolls_everything_back(faulty_conn):
    gid = register()["id"]
    faulty_conn.arm(lambda sql: sql == "COMMIT")
    resp = replace_v2(gid, expected=1, http=no_raise)
    assert resp.status_code == 500
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 1
    assert got["nodes"] == GRAPH_V1["nodes"]
    assert get_store().get_graph_version_row(gid, 2) is None

    resp = replace_v2(gid, expected=1)
    assert resp.status_code == 200
    assert resp.json()["graph_version"] == 2


# ------------------------- concurrency ordering -----------------------------


def test_concurrent_replacements_serialize_one_stales_out():
    gid = register()["id"]
    barrier = threading.Barrier(2)
    outcomes = []

    def replace(version):
        barrier.wait()
        try:
            registry.replace_graph(
                gid, version, GRAPH_V2["nodes"], GRAPH_V2["export"])
            outcomes.append(200)
        except registry.StaleVersion:
            outcomes.append(409)

    t1 = threading.Thread(target=replace, args=(1,))
    t2 = threading.Thread(target=replace, args=(1,))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert sorted(outcomes) == [200, 409]
    # Exactly one version increment happened; the loser changed nothing.
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 2
    assert got["nodes"] == GRAPH_V2["nodes"]


def test_revision_and_registry_replacement_take_effect_in_one_order(
    monkeypatch,
):
    """A revision either fully precedes or fully follows a replacement.

    Hold the store lock inside the revision's review so a registry
    replacement queues behind it: the in-flight revision commits v2 under
    generation 0, then the replacement commits generation 1 and re-reviews the
    latest (v2) graph under the new registry.
    """
    gid = register()["id"]
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

    def do_replace():
        outcomes["revision"] = registry.replace_graph(
            gid, 1, GRAPH_V2["nodes"], GRAPH_V2["export"])

    worker = threading.Thread(target=do_replace)
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

    # The revision committed version 2 at the then-current generation 0 ...
    revision = outcomes["revision"]
    assert revision["id"] == gid
    assert revision["graph_version"] == 2
    assert revision["generation"] == 0
    assert revision["review"]["blocked"] is False
    # ... then the replacement re-reviewed the latest v2 graph. v2 nulls the
    # email content, so it stays safe even with the registry entry.
    assert outcomes["replace"]["generation"] == 1
    assert outcomes["replace"]["reviews"][gid]["blocked"] is False
    got = client.get(f"/api/graphs/{gid}").json()
    assert got["graph_version"] == 2
    assert got["generation"] == 1
    assert got["review"]["blocked"] is False
