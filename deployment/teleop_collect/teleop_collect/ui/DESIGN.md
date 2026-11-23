# ui: design

The operator's page and the small web server behind it (STRUCTURE.md §9, PLAN.md §3, §4.3–4.4, §5.4).

## What it does and does not do

- Serves one offline page (plain HTML/CSS/JS in `static/`, no CDN, no build; the IBM Plex Sans font bundled) on
  `127.0.0.1:<ports.http>` **only**.
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
| `static/` | `index.html`, `app.js`, `style.css`, `theme.js` (sets the theme in `<head>`, before the first paint); `fonts/ibm-plex-sans-latin-wght-normal.woff2` (IBM Plex Sans variable, latin, from `@fontsource-variable/ibm-plex-sans@5.2.8`) and its SIL OFL `fonts/LICENSE.txt` from the same package |

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
  `name` (select_task, create_task) collides with the command's `name`. The commands: start_episode, stop_episode
  `{outcome}`, stop_arm, select_task `{name}`, create_task `{name, description}`, delete_episode `{episode}`,
  label_episode `{episode, outcome}`, delete_task `{name}` (the folder and all its episodes; refused while recording)
  and set_data_root `{path}` (absolute or `~`, writable; refused while recording; on ok the task is deselected and the
  lists reload). The status's `data_root` is the storage folder (the page shows `-` when it is missing).

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
  only while focused. Esc = Stop arm in every phase, even while typing. Keys typed into any field (task form, storage
  folder, delete-task confirmation; all `<input>`, covered by `isTyping`) are not captured, and the key map shows
  the keys off while a field has the focus. The open forms close (their field blurred) when recording starts and on
  Stop arm / Esc. Stop episode opens Save: success / Save: fail / Discard (one `stop_episode`) or Resume recording;
  keys are off meanwhile.
- **The page's layout** (2026-11-22, for a new operator; Apple-like, color only for status): a translucent header
  (a gripper mark and the name, three health dots Arm / Cameras / Link
  with their details in the tooltip and in visually hidden text, the storage folder (shown as it is, cut in the
  middle when long) with an inline Change field, Mock PC pill, the theme toggle), the alert banner,
  then two columns that fit 1440×900 without scrolling (below 1000 px one column: cameras, task + state, Stop arm,
  the keys, then episodes and the log, so the keys stay on screen while recording):
  - left: the cameras by role (main IMX519 = "Wrist camera", ultrawide IMX219 = "Wide view"; `img` ids stay
    `preview-<camera>`), each with a health dot + fps and a "No signal" placeholder; click / Enter enlarges one (the
    other becomes a thumbnail), again restores; a REC m:ss badge and the "At workspace limit" chip over the video.
    Below, the **keycap map**: Move (W/A/S/D/R/F laid out as on the keyboard, directions seen from behind the robot),
    Rotate (U I O over J K L), Gripper (N, M; a dot marks the last command), Shift / Esc; held keys light up in the
    accent color (from the held set the page sends, Shift when slow). When keys are off the map dims and one line says
    why (not connected, start an episode, choose save or discard, click the page, typing in a field). Diagnostics are a closed
    `<details>`.
  - right: the task (select, New task inline form, Delete task…), the **state card** (the one place saying what to do
    next, with its button: Not ready + teleop's `start_refusal`, Ready + Start episode, Recording + timer + Stop
    episode, the save choice; a refused command's reason in orange for 6 s; successes are not echoed, the new state
    is the feedback; a form's refusal shows in the form), Stop arm (red, always enabled while connected), the episodes
    (newest first, counts, outcome pills, one line each; unsaved rows offer Save ✓ / Save ✗ = `label_episode`), and the
    activity log (a closed `<details>`; an orange dot on it when a warning or error arrived since it was last open).
- **Inline confirmations, no browser dialogs:** deleting an episode turns its row into "Delete episode_NNN from disk?
  No / Yes, delete"; deleting a task needs its name typed exactly (the red button is enabled only then). Disabled while
  recording or busy. Discard (in the save choice) asks "Discard episode_NNN? It is deleted from disk." with Yes,
  discard / Back; the save choice stays open (keys off) meanwhile.
- **The state card's line** (outside an episode): when Start is allowed, `at the start pose ✓ ready to start` once the
  PC has accepted a start there (`start_pose_confirmed`), else `at the start-up pose ✓ ready to start (the PC checks
  it at Start)`; when not, teleop's `start_refusal` (e.g. `put the arm at the start pose: h, then p on the PC`). Only
  the first letter is capitalized; the text is teleop's. During an episode: `Hold keys to move · wait ~1 s after
  Start`, or (save choice open) that keys are off until the operator chooses.
- **Offline:** no `http://`/`https://` text in the three files (inline SVG is written without `xmlns`, which HTML does
  not need), no `@import`, every `src`/`href`/`url()` under `/static/`. The font is IBM Plex Sans (chosen by the user
  2026-11-22 over Inter and Barlow; SIL OFL 1.1, license next to it), then the system stack (`-apple-system`,
  `system-ui`, ...); the system monospace stack (no IBM Plex Mono: there is no variable package of it) only for paths,
  episode names and diagnostics.
- **Theme: dark by default**, whatever the OS setting (the user's choice, 2026-11-22). The light tokens stay, behind
  `<html data-theme="light">`: a quiet sun / moon button in the header ("Light theme" / "Dark theme") switches it and
  remembers it in `localStorage` (`teleop_collect.theme`; blocked storage = dark, not remembered). `theme.js` in
  `<head>` applies it before the first paint, so there is no flash; `index.html` starts as `data-theme="dark"`. The
  toggle is an ordinary button (blurred on mouseup like the others), and no key is bound to it.
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
`test_page_static.py` (app.js holds the §4.1 map, the focus rules, offline, the bundled font, every field covered by
`isTyping`, every command known, inline confirmations, no Home button; `node --check` when node exists);
`test_settings_and_process.py` (settings; the process: ready, SIGINT/SIGTERM exit 0,
FATAL lines).

## How to extend

- **A new key:** add its name to `MOTION_KEYS` (E1, contracts) and teleop's handling (E4), then one entry in
  `MOTION_KEY_MAP` and `MOTION_KEY_ORDER` in `app.js`, a keycap (`data-key="<name>"`) in `index.html`, and
  `EXPECTED_MOTION` in `test_page_static.py`.
- **A new button:** a U1 request (E1) served by teleop (E3), one `COMMANDS` entry in `server.py` (its argument names
  and builder), a button calling `command("<name>", {args})` in `app.js`. ScriptedPage needs nothing.
- **A new status field:** it arrives by itself (the status is TeleopStatus as JSON); show it in `render()`.
