# adapted from rnoro/deployment:deployment/contracts/contracts/__init__.py @ 91b178b
# changes: renamed to deploy_contracts; only what crosses between laptop and PC (C2, C3 + hello, ack, error, the yaml
# loaders, compare_hello, SHARED_SCHEMAS); C0/C1/C4, sizes and the C0 multipart helpers moved to policy_deploy.
"""deploy_contracts: the contracts shared by the laptop (policy_deploy) and the PC (control_deploy).

C2 (arm state) and C3 (commands, incl. hello) with their validators, the msgpack wire format, schema names, the three
contract yaml files (limits, frames, link), the example messages and the ZeroMQ helpers. Importing this package
registers exactly SHARED_SCHEMAS. See deployment/STRUCTURE.md section 3.
"""
from __future__ import annotations

from deploy_contracts import paths
from deploy_contracts.c2_state import MODES, ArmState, GateCounters, rigid_problem
from deploy_contracts.c3_commands import (HOME_STOPPED_PREFIX, GetState, Heartbeat, Hello, HelloReply, Home,
                                          SetWaypoints, SetWaypointsReply, Start, Stop, WaypointIssue,
                                          home_was_stopped)
from deploy_contracts.codec import decode, encode, pack_array, pack_obj, unpack_array, unpack_obj
from deploy_contracts.common import Ack, ErrorReply
from deploy_contracts.errors import PROBLEMS, ContractError, RemoteError, TransportCancelled, TransportTimeout
from deploy_contracts.examples import (Example, broken_example, build_examples, drop_key, good_example, load_index,
                                       raw_example, resolve_path, set_key)
from deploy_contracts.frames import TcpFrame
from deploy_contracts.hello_check import HelloMismatch, compare_hello
from deploy_contracts.limits import ContractLimits
from deploy_contracts.link import LinkSettings
from deploy_contracts.message import Message, Record
from deploy_contracts.paths import contracts_dir, examples_dir, frames_file, limits_file, link_file
from deploy_contracts.pubsub import Publisher, Subscriber
from deploy_contracts.registry import lookup, register, registered
from deploy_contracts.rpc import RpcClient, RpcServer
from deploy_contracts.schema import SchemaName
from deploy_contracts.shared import SHARED_SCHEMAS

__all__ = [
    "paths", "contracts_dir", "limits_file", "frames_file", "link_file", "examples_dir",
    "Ack", "ErrorReply",
    "ArmState", "GateCounters", "MODES", "rigid_problem",
    "SetWaypoints", "SetWaypointsReply", "WaypointIssue", "Heartbeat", "Start", "Stop", "Home", "GetState",
    "Hello", "HelloReply", "HOME_STOPPED_PREFIX", "home_was_stopped",
    "SHARED_SCHEMAS", "HelloMismatch", "compare_hello",
    "encode", "decode", "pack_obj", "unpack_obj", "pack_array", "unpack_array",
    "ContractError", "TransportTimeout", "TransportCancelled", "RemoteError", "PROBLEMS",
    "ContractLimits", "TcpFrame", "LinkSettings",
    "Message", "Record", "SchemaName", "register", "lookup", "registered",
    "RpcClient", "RpcServer", "Publisher", "Subscriber",
    "Example", "build_examples", "load_index", "good_example", "broken_example", "raw_example", "set_key", "drop_key",
    "resolve_path",
]
