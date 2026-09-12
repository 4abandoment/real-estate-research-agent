"""Manage the Slack app manifest from the repo (Slack config as code).

Requires an app configuration token (api.slack.com/apps -> Your App
Configuration Tokens). Config tokens expire after 12 hours; use ``rotate`` to
mint a new one with the stored refresh token.

Usage:
    python scripts/slack_manifest.py export
    python scripts/slack_manifest.py validate
    python scripts/slack_manifest.py update
    python scripts/slack_manifest.py rotate
"""

import json
import sys
from pathlib import Path

from slack_sdk import WebClient

from research_agent.config import load_settings

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "slack" / "manifest.json"


def _call(client: WebClient, method: str, *, verb: str = "POST", **params):
    response = client.api_call(method, http_verb=verb, params=params)
    if not response.get("ok"):
        raise SystemExit(f"{method} failed: {response.get('error')}")
    return response.data


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "export"
    settings = load_settings()

    if command == "rotate":
        client = WebClient(token="")
        data = _call(
            client,
            "tooling.tokens.rotate",
            refresh_token=settings.slack_config_refresh_token,
        )
        print(
            json.dumps({"token": data["token"], "refresh_token": data["refresh_token"]}, indent=2)
        )
        return

    if not settings.slack_config_token or not settings.slack_app_id:
        raise SystemExit("SLACK_CONFIG_TOKEN and SLACK_APP_ID are required.")

    client = WebClient(token=settings.slack_config_token)

    if command == "export":
        data = _call(client, "apps.manifest.export", verb="GET", app_id=settings.slack_app_id)
        print(json.dumps(data["manifest"], indent=2))
    elif command == "validate":
        manifest = MANIFEST_PATH.read_text(encoding="utf-8")
        print(_call(client, "apps.manifest.validate", manifest=manifest))
    elif command == "update":
        manifest = MANIFEST_PATH.read_text(encoding="utf-8")
        print(
            _call(client, "apps.manifest.update", app_id=settings.slack_app_id, manifest=manifest)
        )
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
