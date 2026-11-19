"""teleop_collect.ui: the operator's page (STRUCTURE.md 9, PLAN.md 4.3).

A small aiohttp server on 127.0.0.1 only. It serves one offline page, forwards the page's keys and buttons to teleop
(U1) and pulls the previews from frames (U2). It owns nothing; it shows and forwards. See DESIGN.md.
"""
from __future__ import annotations

NAME = "ui"
