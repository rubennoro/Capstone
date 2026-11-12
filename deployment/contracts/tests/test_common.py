from contracts import Ack, ErrorReply, decode, encode
from helpers import contract_error


def test_ack():
    assert decode(encode(Ack(ok=True, reason=""))) == Ack(ok=True, reason="")
    refused = Ack(ok=False, reason="refused: mode is fault")
    assert decode(encode(refused)).reason == "refused: mode is fault"
    contract_error(lambda: Ack(ok=1, reason=""), "ok", "type")  # type: ignore[arg-type]
    contract_error(lambda: Ack(ok=True, reason=None), "reason", "type")  # type: ignore[arg-type]


def test_error_reply():
    e = ErrorReply(request="c4.infer/v1", error="c4.infer/v1: field 'main': shape (2, 224, 3), expected "
                   "(2, 224, 224, 3)", field="main")
    back = decode(encode(e))
    assert (back.request, back.field) == ("c4.infer/v1", "main") and back.error == e.error
    assert list(e.to_dict()) == ["schema", "request", "error", "field"]
    contract_error(lambda: ErrorReply(request="", error="x", field=3), "field", "type")  # type: ignore[arg-type]
