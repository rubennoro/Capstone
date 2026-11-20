# ui: design

The operator's page and the small web server behind it (STRUCTURE.md §9, PLAN.md §3, §4.3–4.4, §5.4).

## What it does and does not do

- Serves one offline page (plain HTML/CSS/JS in `static/`, no CDN, no build) on `127.0.0.1:<ports.http>` **only**.
- Forwards the page's key state, finger commands and buttons to teleop (U1), and pushes teleop's status to the page.
- Pulls the newest JPEG of each camera from frames (U2) and pushes it to the page.
- Enforces a **deadman** on the key state.
- It owns nothing and decides nothing: refusals (Start away from the start pose, Start while the PC is homing, ...)
  come from teleop's acks and status. It never moves the arm to a pose: there is no Home button and no `home`
  command; the arm is homed (`h`) and put at the start pose (`p`) with the PC's keys (since 2026-11-20, as
  final-deployment @ 927a5ba). It never talks to the PC or the Pi, and imports no other process subpackage.
- It never exits on its own when teleop or frames are slow or gone: the page shows `teleop not answering` /
  `frames not answering`. It ends on SIGINT/SIGTERM (exit 0) or a FATAL (a reply breaking its contract, the HTTP
  port taken, bad settings; exit 1 with one `FATAL ui: ...` line).

## Files

| File | Job |
|---|---|
| `__main__.py` | `python -m teleop_collect.ui --station S [--session D] [--ready-file F] [--mock]` (pd:frames/__main__.py pattern: ProcessArgs, print_fatal, signals) |
| `settings.py` | `Settings`: config/ui.yaml + station `ui:`, validated; endpoints and the HTTP port from the station |
| `bridge.py` | `U1Client`, `U2Client` (RpcClient, one REQ socket per thread), `Bridge` (awaitable calls on lanes), `FatalError` |
| `server.py` | `UiServer` (HTTP, WebSocket, pollers, `KeysArbiter` + deadman, `KeysSender`), `UiApp` (the process: loop, ready file, stop) |
| `scripted_page.py` | `ScriptedPage`: the page's protocol from Python, synchronous, for tests (E7) |
| `static/` | `index.html`, `app.js`, `style.css` |

## Threads and data flow

One asyncio event loop (the main thread) runs the HTTP server and every task. Each U1/U2 request runs on a
**lane**: a one-thread executor with its own REQ socket (ZeroMQ sockets are not thread-safe; a REQ carries one request
at a time). So the loop never blocks, and slow traffic of one kind never delays another:

| Lane | Traffic |
|---|---|
| status | `u1.get_status(after_message)` at `status_hz` → one status message to every page |
| keys | `u1.keys`, strictly in order; `KeysSender` coalesces (the newest state replaces one not yet sent) |
| command | `u1.gripper` and the buttons |
| stop | `u1.stop_arm` alone, so Stop arm / Esc never queues behind another button |
| preview | `u2.get_preview(main_after, ultrawide_after)` at `preview_hz`, only while a page is connected |

Each page has an outbox and a writer task: a preview header and its JPEG go out back to back; status and previews
are dropped when a page is 6 sends behind (acks never are).

## WebSocket protocol (STRUCTURE.md §9.2, with three additions)

- server → page: `hello` once on connect (`keys_resend_ms`, `keys_stale_ms`, `status_hz`, `preview_hz`,
  `motion_keys`) — **added**, so the page and ScriptedPage use the configured resend period.
- server → page: `status` = every TeleopStatus field (JSON; `tcp_m` a list), plus the ui's link fields
  `teleop_answering`, `teleop_error`, `frames_answering`, `frames_error`, `preview_health` — **added**. Before teleop
  ever answered, only the link fields. `messages` holds the newest 50 the ui has seen (pages keep their own by id).
- server → page: `preview` header then one binary message (the JPEG exactly as frames sent it).
- server → page: `ack {id, ok, reason}`; `error {text}` for a message the server could not use — **added**.
- page → server: `keys {held, slow, focus}`, `gripper {command}`, and
  `command {name, args: {...}, id}` — **changed**: the arguments travel in their own `args` map, because a flat
  `name` (select_task, create_task) collides with the command's `name`.

## Decisions

- **Deadman** (PLAN §5.1, §7 "the ui stopped sending keys"): the page that last sent `focus:true` drives the keys.
  When it is silent for `keys_stale_ms`, closes its socket, sends `focus:false` or a broken key state, the server sends
  `keys(held=[], slow=false, focus=false)` at once and no page drives until one sends `focus:true` again. A `focus:false`
  from a page that is not driving changes nothing. Several pages: **the last one with focus wins**. teleop's own
  `key_stale_ms` is the second line. On shutdown the server releases the keys before closing.
- **The page** (PLAN §4.3, §5.4): keys by `KeyboardEvent.code` (layout independent), the map of STRUCTURE §4.1.
  Auto-repeat ignored; keydown/keyup maintain the held set; Shift = slow. Motion and N/M count only while the page has
  focus, teleop answers, the phase is `recording` and the save choice is not open; a key held before that must be
  pressed again. `blur` and a hidden tab send `held:[], focus:false` at once; the resend (every `keys_resend_ms`) runs
  only while focused. Esc = Stop arm in every phase, even while typing. Keys typed into the task form are not captured.
  Stop episode opens Save: success / Save: fail / Discard (one `stop_episode`) or Keep recording; keys are off
  meanwhile. Delete asks `confirm()`.
- **The hint under Start** (outside an episode): when Start is allowed, `at the start pose ✓ ready to start` once the
  PC has accepted a start there (`start_pose_confirmed`), else `at the start-up pose ✓ ready to start (the PC checks
  it at Start)`; when not, teleop's `start_refusal` as it is (e.g. `put the arm at the start pose: h, then p on the
  PC`). During an episode: `recording: hold keys to move`, or `motion off: save or discard the episode`.
- **Local only:** bound on 127.0.0.1; plus a Host check (DNS rebinding) and an Origin check (browsers let any site
  open a WebSocket to localhost), so only a page opened as `127.0.0.1`/`localhost` on this port can drive the arm.
  An `Origin` header must be exactly `http://127.0.0.1:<port>` or `http://localhost:<port>`; anything else is
  refused (403), including `Origin: null` (a sandboxed iframe or a `file://` page), another scheme or another port.
  A request without an `Origin` header (ScriptedPage, curl) is allowed: browsers always send one on a WebSocket.
- **Ready** = the HTTP socket bound and teleop answered one `get_status` (the status poller starts only after the bind).
- **Failures** (PLAN §7): a timeout or an error/v1 reply = "not answering" / its text, and the ui keeps going; a reply
  that breaks its contract (U1 or U2) is a FATAL (conclude: "a message failing its contract").
- If a camera's `frame_no` goes back (the Pi stream restarted), the ui asks from -1 again.

## Tests (`tests/ui/`, helpers `tests/ui_helpers.py`)

Fake U1 and U2 RpcServers on free ports, the server on HTTP port 0 or a free port, no browser:
`test_server.py` (127.0.0.1-only bind via /proc/net/tcp, port in use, the page and static files, Host/Origin checks,
ready rule, teleop/frames gone, status rate and fields, messages, every command and its ack, refusals, bad commands,
the stop lane, previews only newer, the deadman on silence / close / blur / bad keys, multi-page focus, gripper,
shutdown release, contract break = fatal); `test_scripted_page.py` (round trips E7 uses, `wait_start_allowed()`);
`test_page_static.py` (app.js holds the §4.1 map, the focus rules, offline, every command known, no Home button;
`node --check` when node exists); `test_settings_and_process.py` (settings; the process: ready, SIGINT/SIGTERM exit 0,
FATAL lines).

## How to extend

- **A new key:** add its name to `MOTION_KEYS` (E1, contracts) and teleop's handling (E4), then one entry in
  `MOTION_KEY_MAP` and `MOTION_KEY_ORDER` in `app.js`, the legend in `index.html`, and `EXPECTED_MOTION` in
  `test_page_static.py`.
- **A new button:** a U1 request (E1) served by teleop (E3), one `COMMANDS` entry in `server.py` (its argument names
  and builder), a button calling `command("<name>", {args})` in `app.js`. ScriptedPage needs nothing.
- **A new status field:** it arrives by itself (the status is TeleopStatus as JSON); show it in `render()`.
