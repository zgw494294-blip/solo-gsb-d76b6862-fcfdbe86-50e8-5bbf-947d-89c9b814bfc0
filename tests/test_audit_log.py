"""Audit log of committed registry replacements and sample purges.

Both kinds of committed operations append one row to a shared, append-only
audit log (``GET /api/audit-log``) with a single positive ``seq`` that grows
in the order operations take effect:

* a registry replacement records ``generation_before``/``generation_after``
  and the source columns it ``added``/``removed`` (each sorted by
  ``(source, column)``); a replacement with identical contents still counts;
* a purge records the current ``generation``, the requested source
  ``columns`` (sorted) and the actually purged ``package_ids`` (sorted).

Records never contain sample rows or whole graphs; stale (409) replacements
and purges that match nothing leave no record. Reads page over
``after_seq < seq <= until_seq`` ascending; omitting ``until_seq`` pins the
cutoff to the largest committed seq while an explicit cutoff above it is a
400, and an empty page carries ``after_seq`` forward as the next cursor.
"""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage import get_store, reset_store

client = TestClient(app)
# A storage fault must surface as a 500/INTERNAL response, not a test crash.
no_raise = TestClient(app, raise_server_exceptions=False)

GRAPH = {
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
ROWS = {
    "users": [
        {"uid": 1, "email": "a@x.example", "country": "CN"},
        {"uid": 2, "email": "b@x.example", "country": "US"},
    ]
}
CONDITIONS = {"only_cn": {"country": "CN"}}
ENTRIES = [
    {"source": "users", "column": "email"},
    {"source": "users", "column": "country"},
]


def register(payload=GRAPH):
    resp = client.post("/api/graphs", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def seal_ok(gid, rows=ROWS, conditions=CONDITIONS):
    resp = client.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": rows, "conditions": conditions},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def put_registry(entries, generation, http=None):
    return (http or client).put(
        "/api/sensitive-columns",
        json={"entries": entries, "expected_generation": generation},
    )


def purge(*columns, http=None):
    return (http or client).post(
        "/api/packages/purge",
        json={"columns": [{"source": s, "column": c} for s, c in columns]},
    )


def audit(**params):
    return client.get("/api/audit-log", params=params)


def all_records():
    return audit().json()["records"]


def reopen_database():
    reset_store(get_store()._path)


# --------------------------------------------------------------- empty log --


def test_empty_log_defaults_to_cutoff_zero():
    body = audit().json()
    assert body == {"until_seq": 0, "records": [], "next_cursor": 0}


# --------------------------------------------------------- replace records --


def test_replace_records_generations_and_added_columns():
    assert put_registry(ENTRIES, 0).status_code == 200
    records = all_records()
    assert len(records) == 1
    assert records[0] == {
        "seq": 1,
        "type": "replace",
        "generation_before": 0,
        "generation_after": 1,
        # Sorted by (source, column): country before email, regardless of
        # submission order.
        "added": [
            {"source": "users", "column": "country"},
            {"source": "users", "column": "email"},
        ],
        "removed": [],
    }


def test_replace_records_removed_columns_sorted():
    assert put_registry(
        [{"source": "users", "column": "email"},
         {"source": "pay", "column": "card"}], 0
    ).status_code == 200
    assert put_registry(
        [{"source": "users", "column": "email"}], 1
    ).status_code == 200
    record = all_records()[-1]
    assert record["generation_before"] == 1
    assert record["generation_after"] == 2
    assert record["added"] == []
    assert record["removed"] == [{"source": "pay", "column": "card"}]


def test_replace_keeps_added_and_removed_in_one_record():
    assert put_registry(
        [{"source": "a", "column": "x"}], 0
    ).status_code == 200
    assert put_registry(
        [{"source": "b", "column": "y"}], 1
    ).status_code == 200
    record = all_records()[-1]
    assert record["added"] == [{"source": "b", "column": "y"}]
    assert record["removed"] == [{"source": "a", "column": "x"}]


def test_identical_content_replace_is_still_recorded():
    assert put_registry(ENTRIES, 0).status_code == 200
    # Same set, different submission order: nothing added or removed, the
    # generation still advances and one record is still appended.
    assert put_registry(list(reversed(ENTRIES)), 1).status_code == 200
    records = all_records()
    assert [r["seq"] for r in records] == [1, 2]
    assert records[1]["generation_before"] == 1
    assert records[1]["generation_after"] == 2
    assert records[1]["added"] == []
    assert records[1]["removed"] == []


def test_stale_replacement_leaves_no_record():
    assert put_registry(ENTRIES, 0).status_code == 200
    resp = put_registry([], 7)
    assert resp.status_code == 409
    records = all_records()
    assert len(records) == 1
    assert records[0]["seq"] == 1
    # The generation did not move either, so the rejected expectation stays
    # stale on retry.
    assert put_registry([], 7).status_code == 409
    assert len(all_records()) == 1


def test_clear_replace_records_removed_columns():
    assert put_registry(ENTRIES, 0).status_code == 200
    assert put_registry([], 1).status_code == 200
    record = all_records()[-1]
    assert record["added"] == []
    assert record["removed"] == [
        {"source": "users", "column": "country"},
        {"source": "users", "column": "email"},
    ]


# ----------------------------------------------------------- purge records --


def test_purge_records_generation_requested_columns_and_package_ids():
    gid = register()
    pid = seal_ok(gid)
    # Submit the request columns unsorted; the record sorts by (source, col).
    resp = purge(("users", "email"), ("ghost", "x"))
    assert resp.status_code == 200
    assert resp.json() == {"purged": [pid]}
    records = all_records()
    assert len(records) == 1
    assert records[0] == {
        "seq": 1,
        "type": "purge",
        "generation": 0,
        "columns": [
            {"source": "ghost", "column": "x"},
            {"source": "users", "column": "email"},
        ],
        "package_ids": [pid],
    }


def test_empty_match_purge_leaves_no_record():
    gid = register()
    pid = seal_ok(gid)
    resp = purge(("ghost", "x"), ("users", "uid_nope"))
    assert resp.json() == {"purged": []}
    assert all_records() == []
    # The unmatched package is untouched and remains claimable after release.
    assert client.post(
        f"/api/packages/{pid}/release",
        json={"expected_generation": 0},
    ).status_code == 200
    assert client.get(f"/api/packages/{pid}").status_code == 200
    assert all_records() == []


def test_purge_record_package_ids_sorted_and_unique():
    gid = register()
    pid1 = seal_ok(gid)
    pid2 = seal_ok(gid)
    resp = purge(("users", "country"))
    assert resp.json() == {"purged": sorted([pid1, pid2])}
    record = all_records()[-1]
    assert record["package_ids"] == sorted([pid1, pid2])
    # Re-purging the same dependency matches no live packages: no new record.
    assert purge(("users", "country")).json()["purged"] == []
    assert [r["seq"] for r in all_records()] == [1]


def test_purge_record_carries_current_generation_after_replacements():
    gid = register()
    # Two successful replacements advance the registry to generation 2.
    assert put_registry(ENTRIES, 0).status_code == 200
    assert put_registry([], 1).status_code == 200
    pid = seal_ok(gid)
    assert purge(("users", "email")).status_code == 200
    record = all_records()[-1]
    assert record["type"] == "purge"
    assert record["generation"] == 2
    assert record["package_ids"] == [pid]


# --------------------------------------------------- shared seq / ordering --


def test_both_kinds_share_one_increasing_effect_order_seq():
    gid = register()
    pid = seal_ok(gid)
    assert put_registry(ENTRIES, 0).status_code == 200       # seq 1
    assert put_registry([], 1).status_code == 200            # seq 2
    assert purge(("users", "email")).status_code == 200      # seq 3
    # Nothing purged here (package already purged): no seq 4.
    assert purge(("users", "email")).json()["purged"] == []
    assert put_registry(ENTRIES, 2).status_code == 200       # seq 4

    records = all_records()
    assert [r["seq"] for r in records] == [1, 2, 3, 4]
    assert [r["type"] for r in records] == [
        "replace", "replace", "purge", "replace"
    ]
    assert records[2]["package_ids"] == [pid]


def test_records_contain_neither_samples_nor_graphs():
    gid = register()
    pid = seal_ok(gid)
    assert purge(("users", "email")).status_code == 200
    raw = json.dumps(all_records())
    # The sealed row values and the graph's column values must not leak.
    assert "a@x.example" not in raw
    assert "export_users" not in raw
    assert "only_cn" not in raw
    assert pid in raw  # the package id itself is meant to be traceable
    assert gid not in raw


# ------------------------------------------------------------- pagination --


def _make_operations(n):
    generation = client.get("/api/sensitive-columns").json()["generation"]
    for _ in range(n):
        assert put_registry([], generation).status_code == 200
        generation += 1


def test_pagination_limit_and_cursor():
    _make_operations(5)
    page1 = audit(limit=2).json()
    assert page1["until_seq"] == 5
    assert [r["seq"] for r in page1["records"]] == [1, 2]
    assert page1["next_cursor"] == 2

    page2 = audit(after_seq=page1["next_cursor"], limit=2).json()
    assert [r["seq"] for r in page2["records"]] == [3, 4]
    assert page2["next_cursor"] == 4

    page3 = audit(after_seq=4, limit=2).json()
    assert [r["seq"] for r in page3["records"]] == [5]
    assert page3["next_cursor"] == 5


def test_empty_page_cursor_carries_after_seq_forward():
    _make_operations(2)
    body = audit(after_seq=2, limit=10).json()
    assert body["records"] == []
    assert body["next_cursor"] == 2


def test_default_limit_is_fifty_and_max_one_hundred():
    _make_operations(3)
    assert len(audit().json()["records"]) == 3  # default does not error
    assert audit(limit=100).status_code == 200


def test_window_is_strict_after_and_inclusive_until():
    _make_operations(4)
    body = audit(after_seq=1, until_seq=3).json()
    assert body["until_seq"] == 3
    assert [r["seq"] for r in body["records"]] == [2, 3]


def test_fixed_until_seq_never_mixes_in_later_operations():
    _make_operations(2)
    # Fix the cutoff at 2 and page with limit 1.
    page1 = audit(after_seq=0, limit=1, until_seq=2).json()
    assert [r["seq"] for r in page1["records"]] == [1]
    assert page1["until_seq"] == 2
    page2 = audit(after_seq=1, limit=1, until_seq=2).json()
    assert [r["seq"] for r in page2["records"]] == [2]
    assert page2["until_seq"] == 2
    page3 = audit(after_seq=2, limit=1, until_seq=2).json()
    assert page3["records"] == []
    assert page3["until_seq"] == 2

    # New operations take effect after the cutoff was fixed ...
    _make_operations(2)  # seqs 3 and 4
    # ... replaying the fixed window still shows nothing past seq 2.
    replay = audit(after_seq=0, limit=10, until_seq=2).json()
    assert [r["seq"] for r in replay["records"]] == [1, 2]
    assert replay["until_seq"] == 2


def test_omitted_until_seq_follows_latest_committed_seq():
    _make_operations(1)
    assert audit().json()["until_seq"] == 1
    _make_operations(2)  # generation starts at 1: seqs 2 and 3
    assert audit().json()["until_seq"] == 3


def test_explicit_until_seq_equal_to_max_is_accepted():
    _make_operations(2)
    body = audit(until_seq=2).json()
    assert body["until_seq"] == 2
    assert [r["seq"] for r in body["records"]] == [1, 2]


def test_until_seq_above_max_is_rejected():
    _make_operations(1)
    resp = audit(until_seq=2)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "UNTIL_SEQ_OUT_OF_RANGE"
    assert error["until_seq"] == 2
    assert error["max_seq"] == 1
    # The cutoff above the maximum is rejected on an empty log too, where the
    # only legal explicit cutoff is 0.
    reset_store(str(get_store()._path) + ".fresh")
    assert audit(until_seq=1).status_code == 400
    assert audit(until_seq=0).status_code == 200


def test_after_seq_beyond_until_seq_returns_empty_page():
    _make_operations(3)
    body = audit(after_seq=3, until_seq=2).json()
    assert body == {"until_seq": 2, "records": [], "next_cursor": 3}


def test_query_parameter_validation():
    assert audit(after_seq=-1).status_code == 422
    assert audit(limit=0).status_code == 422
    assert audit(limit=101).status_code == 422
    assert audit(until_seq=-1).status_code == 422
    assert audit(after_seq="x").status_code == 422


# ---------------------------------------------------------- persistence -----


def test_log_survives_restart():
    gid = register()
    pid = seal_ok(gid)
    assert put_registry(ENTRIES, 0).status_code == 200
    assert put_registry([], 1).status_code == 200
    assert purge(("users", "email")).status_code == 200

    reopen_database()
    body = audit().json()
    assert body["until_seq"] == 3
    assert [r["seq"] for r in body["records"]] == [1, 2, 3]
    assert [r["type"] for r in body["records"]] == [
        "replace", "replace", "purge"]
    assert body["records"][2]["package_ids"] == [pid]
    # New operations continue the seq sequence after reopening.
    assert put_registry(ENTRIES, 2).status_code == 200
    assert audit().json()["records"][-1]["seq"] == 4


def test_old_database_starts_with_empty_log():
    # A database written before the audit feature existed: meta only, no
    # audit table. Opening it must not backfill anything.
    path = str(get_store()._path) + ".legacy"
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        con.execute("INSERT INTO meta (key, value) VALUES ('generation', '5')")
        con.commit()
    finally:
        con.close()

    reset_store(path)
    body = audit().json()
    assert body == {"until_seq": 0, "records": [], "next_cursor": 0}
    # The first post-upgrade replacement starts the log at seq 1 and keeps the
    # pre-existing generation.
    assert put_registry([], 5).status_code == 200
    records = all_records()
    assert len(records) == 1
    assert records[0]["seq"] == 1
    assert records[0]["generation_before"] == 5
    assert records[0]["generation_after"] == 6


# ------------------------------------------------- atomicity with business --


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


@pytest.fixture
def faulty_conn():
    store = get_store()
    real = store._conn
    proxy = FaultyConnection(real)
    store._conn = proxy
    yield proxy
    store._conn = real


def test_replace_audit_row_rolls_back_together_with_the_change(faulty_conn):
    register()
    faulty_conn.arm(lambda sql: sql == "COMMIT")
    resp = put_registry(ENTRIES, 0, http=no_raise)
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "INTERNAL"

    # Neither the business change nor the audit row survived.
    assert client.get("/api/sensitive-columns").json() == {
        "generation": 0, "entries": []}
    assert all_records() == []

    reopen_database()
    assert client.get("/api/sensitive-columns").json() == {
        "generation": 0, "entries": []}
    assert audit().json()["next_cursor"] == 0

    # The retried replacement takes seq 1 — no seq was burned by the failure.
    assert put_registry(ENTRIES, 0).status_code == 200
    assert all_records()[0]["seq"] == 1


def test_replace_audit_insert_failure_leaves_both_states_untouched(faulty_conn):
    register()
    faulty_conn.arm(lambda sql: sql.startswith("INSERT INTO audit_log"))
    resp = put_registry(ENTRIES, 0, http=no_raise)
    assert resp.status_code == 500
    assert client.get("/api/sensitive-columns").json() == {
        "generation": 0, "entries": []}
    assert all_records() == []
    # Same expected generation still works, log starts at seq 1.
    assert put_registry(ENTRIES, 0).status_code == 200
    assert all_records()[0]["seq"] == 1


def test_purge_audit_row_rolls_back_together_with_the_purge(faulty_conn):
    gid = register()
    pid = seal_ok(gid)
    faulty_conn.arm(lambda sql: sql == "COMMIT")
    resp = purge(("users", "email"), http=no_raise)
    assert resp.status_code == 500

    # No package was purged and no audit row survived.
    assert client.post(
        f"/api/packages/{pid}/release",
        json={"expected_generation": 0},
    ).status_code == 200
    assert client.get(f"/api/packages/{pid}").status_code == 200
    assert all_records() == []

    reopen_database()
    assert all_records() == []
    # The retried purge takes seq 1.
    assert purge(("users", "email")).status_code == 200
    assert all_records()[0] == {
        "seq": 1,
        "type": "purge",
        "generation": 0,
        "columns": [{"source": "users", "column": "email"}],
        "package_ids": [pid],
    }
    assert client.get(f"/api/packages/{pid}").status_code == 410
