"""SHARED_SCHEMAS (STRUCTURE.md 3.4): sorted, unique, equal to the schemas of the classes deploy_contracts registers."""
from __future__ import annotations

import deploy_contracts
from deploy_contracts import SHARED_SCHEMAS, Message, SchemaName
from deploy_contracts.registry import registered

EXPECTED = ("ack/v1", "c2.state/v1", "c3.get_state/v1", "c3.heartbeat/v1", "c3.hello.reply/v1", "c3.hello/v1",
            "c3.home/v1", "c3.set_waypoints.reply/v1", "c3.set_waypoints/v1", "c3.start/v1", "c3.stop/v1", "error/v1")


def test_exactly_the_twelve():
    assert SHARED_SCHEMAS == EXPECTED
    assert isinstance(SHARED_SCHEMAS, tuple) and len(SHARED_SCHEMAS) == 12


def test_sorted_by_byte_order_and_unique():
    encoded = [s.encode("utf-8") for s in SHARED_SCHEMAS]
    assert encoded == sorted(encoded)
    assert len(set(SHARED_SCHEMAS)) == len(SHARED_SCHEMAS)
    for s in SHARED_SCHEMAS:
        SchemaName.parse(s)  # each well-formed


def test_equal_to_the_registered_shared_classes():
    shared_classes = [c for c in registered() if c.__module__.startswith("deploy_contracts.")]
    assert sorted(c.SCHEMA for c in shared_classes) == list(SHARED_SCHEMAS)


def test_every_exported_message_class_is_shared():
    exported = {getattr(deploy_contracts, n) for n in deploy_contracts.__all__}
    classes = {c for c in exported if isinstance(c, type) and issubclass(c, Message) and c is not Message}
    assert sorted(c.SCHEMA for c in classes) == list(SHARED_SCHEMAS)
