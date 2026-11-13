# adapted from rnoro/deployment:deployment/contracts/tests/test_common.py @ 91b178b
# changes: imports renamed contracts -> deploy_contracts; the ErrorReply case uses a C3 request
from deploy_contracts import Ack, ErrorReply, decode, encode
from helpers import contract_error


def test_ack():
    assert decode(encode(Ack(ok=True, reason=""))) == Ack(ok=True, reason="")
    refused = Ack(ok=False, reason="refused: mode is fault")
    assert decode(encode(refused)).reason == "refused: mode is fault"
    contract_error(lambda: Ack(ok=1, reason=""), "ok", "type")  # type: ignore[arg-type]
    contract_error(lambda: Ack(ok=True, reason=None), "reason", "type")  # type: ignore[arg-type]


def test_error_reply():
    e = ErrorReply(request="c3.set_waypoints/v1", error="c3.set_waypoints/v1: field 'O_T_TCP': shape (8, 3, 4), "
                   "expected (N, 4, 4)", field="O_T_TCP")
    back = decode(encode(e))
    assert (back.request, back.field) == ("c3.set_waypoints/v1", "O_T_TCP") and back.error == e.error
    assert list(e.to_dict()) == ["schema", "request", "error", "field"]
    contract_error(lambda: ErrorReply(request="", error="x", field=3), "field", "type")  # type: ignore[arg-type]


def test_stopped_prefix_is_an_ordinary_reason():
    # STRUCTURE.md 3.7.2: a stopped homing replies ok false with a reason starting "stopped: "; the contract carries it
    a = decode(encode(Ack(ok=False, reason="stopped: the PC's s key stopped the homing move")))
    assert a.ok is False and a.reason.startswith("stopped: ")
