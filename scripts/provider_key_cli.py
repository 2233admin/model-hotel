#!/usr/bin/env python3
"""Provider key CLI for Model Hotel.

This script supports:
1. Import providers from lines in format: account---password---emailjwt---apikey
2. Trigger provider discovery (`discover-all`)
3. Run smoke tests through chat completion endpoint
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional


PROVIDER_PRESET_BASE = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

DEFAULT_MODELS = [
    "deepseek-ai/deepseek-v4-flash",
    "minimaxai/minimax-m3",
]

SENSITIVE_KEYS = {"api_key", "apiKey", "key", "api-key", "authorization"}


@dataclass
class KeyRecord:
    account: str
    password: str
    email_jwt: str
    api_key: str


class ApiError(RuntimeError):
    pass


class ModelHotelClient:
    """Small HTTP client for Model Hotel admin API."""

    def __init__(self, base_api: str, admin_token: str, timeout: int = 30):
        self.base_api = base_api.rstrip("/")
        self.admin_token = admin_token
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> tuple[int, Any]:
        data: Optional[bytes] = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            urllib.parse.urljoin(self.base_api + "/", path.lstrip("/")),
            data=data,
            method=method.upper(),
        )
        req.add_header("Authorization", f"Bearer {self.admin_token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                status = resp.status
        except urllib.error.HTTPError as err:
            raw = err.read().decode("utf-8") if hasattr(err, "read") else ""
            raise ApiError(f"HTTP {err.code}: {raw or err.reason}")
        except urllib.error.URLError as err:
            raise ApiError(f"Request failed: {err}")

        try:
            body = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            body = raw
        return status, body

    def list_providers(self) -> list[dict[str, Any]]:
        status, payload = self._request("GET", "/api/providers")
        if not (200 <= status < 300):
            raise ApiError(f"list providers failed status={status} payload={payload}")

        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                return data
        if isinstance(payload, list):
            return payload

        raise ApiError(f"Unexpected /api/providers response: {payload}")

    def create_provider(self, payload: dict[str, Any]) -> tuple[bool, int, Any]:
        try:
            status, payload_json = self._request("POST", "/api/providers", payload)
            return 200 <= status < 300, status, payload_json
        except ApiError as err:
            msg = str(err)
            if msg.startswith("HTTP "):
                try:
                    status = int(msg.split("HTTP ", 1)[1].split(":", 1)[0])
                    body = msg.split(":", 1)[1].strip() if ":" in msg.split("HTTP ", 1)[1] else msg
                    return False, status, body
                except Exception:
                    return False, 500, msg
            return False, 500, msg

    def discover_all(self) -> tuple[int, Any]:
        try:
            return self._request("POST", "/api/providers/discover-all")
        except ApiError as err:
            return 500, str(err)

    def chat_completion(self, provider: str, model: str, prompt: str) -> tuple[bool, int, Any, float]:
        request_payload = {
            "model": f"{provider}/{model}",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
        }
        start = time.time()
        try:
            status, payload = self._request("POST", "/api/chat/completions", request_payload)
            ok = 200 <= status < 300
            return ok, status, payload, time.time() - start
        except ApiError as err:
            latency = time.time() - start
            msg = str(err)
            if msg.startswith("HTTP "):
                try:
                    status = int(msg.split("HTTP ", 1)[1].split(":", 1)[0])
                    return False, status, msg, latency
                except Exception:
                    return False, 500, msg, latency
            return False, 500, msg, latency


def parse_records(text: str, sep: str, strict: bool) -> Iterator[KeyRecord]:
    for line_no, line in enumerate(text.splitlines(), 1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue

        parts = [p.strip() for p in raw.split(sep)]
        if len(parts) != 4:
            if strict:
                raise ValueError(f"Line {line_no}: expected 4 fields but got {len(parts)}")
            continue

        account, password, email_jwt, apikey = parts
        if not account or not apikey:
            if strict:
                raise ValueError(f"Line {line_no}: account and apikey cannot be empty")
            continue

        yield KeyRecord(account=account, password=password, email_jwt=email_jwt, api_key=apikey)


def dedupe(records: Iterable[KeyRecord]) -> Iterator[KeyRecord]:
    seen: set[str] = set()
    for record in records:
        if record.account in seen:
            continue
        seen.add(record.account)
        yield record


def normalize_provider_name(account: str, suffix: str) -> str:
    name = account.strip()
    return name if name.endswith(suffix) else f"{name}{suffix}"


def sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    for key in list(out.keys()):
        if key in SENSITIVE_KEYS:
            out[key] = "***redacted***"
    return out


def build_payload_variants(
    provider_name: str,
    base_url: str,
    api_key: str,
    enabled: bool,
    autodiscovery: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "name": provider_name,
            "base_url": base_url,
            "api_key": api_key,
            "provider_type": "openai",
            "enabled": enabled,
            "autodiscovery_enabled": autodiscovery,
        },
        {
            "name": provider_name,
            "base_url": base_url,
            "api_key": api_key,
            "type": "openai",
            "enabled": enabled,
        },
        {
            "name": provider_name,
            "baseUrl": base_url,
            "apiKey": api_key,
            "providerType": "openai",
            "enabled": enabled,
            "autodiscoveryEnabled": autodiscovery,
        },
        {
            "name": provider_name,
            "base_url": base_url,
            "api_key": api_key,
            "enabled": enabled,
        },
    ]


def read_input(file_path: str | None) -> str:
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def cmd_import(args: argparse.Namespace, client: ModelHotelClient) -> int:
    text = read_input(args.file)
    records = list(parse_records(text, args.sep, args.strict))
    if args.dedup:
        records = list(dedupe(records))

    extra_payload: Optional[dict[str, Any]] = None
    if args.extra_payload:
        with open(args.extra_payload, "r", encoding="utf-8") as f:
            extra_payload = json.load(f)
            if not isinstance(extra_payload, dict):
                raise ValueError("--extra-payload must be a JSON object")

    existing_names = {p.get("name") for p in client.list_providers()} if args.dup_policy != "overwrite" else set()
    result = {"created": [], "skipped": [], "failed": []}
    enabled = not args.disabled

    for record in records:
        provider_name = normalize_provider_name(record.account, args.suffix)

        if provider_name in existing_names and args.dup_policy == "skip":
            result["skipped"].append({"provider": provider_name, "reason": "already_exists"})
            continue
        if provider_name in existing_names and args.dup_policy == "fail":
            result["failed"].append({"provider": provider_name, "reason": "already_exists"})
            continue

        variants = build_payload_variants(
            provider_name=provider_name,
            base_url=args.base_url,
            api_key=record.api_key,
            enabled=enabled,
            autodiscovery=not args.no_autodiscovery,
        )

        if extra_payload:
            for variant in variants:
                variant.update(extra_payload)

        if args.dry_run:
            result["created"].append(
                {
                    "provider": provider_name,
                    "status": "dry-run",
                    "payload": sanitize_payload(variants[0]),
                }
            )
            continue

        created = False
        for variant in variants:
            ok, status, response = client.create_provider(variant)
            if ok:
                existing_names.add(provider_name)
                result["created"].append({"provider": provider_name, "status": status})
                created = True
                break

            if status not in (400, 422):
                result["failed"].append(
                    {
                        "provider": provider_name,
                        "status": status,
                        "response": response,
                        "payload": sanitize_payload(variant),
                    }
                )
                break

        if not created and not any(item["provider"] == provider_name for item in result["failed"]):
            result["failed"].append({"provider": provider_name, "status": "validation_failed"})

        if args.sleep_ms:
            time.sleep(args.sleep_ms / 1000)

    created_count = len(result["created"])
    skipped_count = len(result["skipped"])
    failed_count = len(result["failed"])
    print(f"import finished: created={created_count}, skipped={skipped_count}, failed={failed_count}")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for fail in result["failed"]:
            print(f"fail: {fail['provider']} -> {fail['status']}")

    return 0 if failed_count == 0 else 2


def cmd_discover(args: argparse.Namespace, client: ModelHotelClient) -> int:
    status, payload = client.discover_all()
    if status >= 300:
        print(f"discover-all failed: status={status}")
        print(payload)
        return 1

    print(f"discover-all success: status={status}")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if isinstance(payload, dict):
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(payload)
    return 0


def cmd_smoke(args: argparse.Namespace, client: ModelHotelClient) -> int:
    providers = client.list_providers()
    provider_map = {p["name"]: p for p in providers if isinstance(p, dict) and "name" in p}

    targets = args.providers
    if not targets:
        targets = list(provider_map.keys())
    else:
        expanded: list[str] = []
        for t in targets:
            expanded.extend([x.strip() for x in t.split(",") if x.strip()])
        targets = expanded

    if not targets:
        print("No providers available")
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("No models configured")
        return 1

    pause = args.pause_ms / 1000 if args.pause_ms else 0.0
    results: list[dict[str, Any]] = []
    fail_count = 0

    for provider in targets:
        if provider not in provider_map:
            fail_count += 1
            results.append({"provider": provider, "status": "not_found"})
            continue

        ok_for_provider = False
        for model in models:
            ok, status, _response, latency = client.chat_completion(provider, model, args.prompt)
            if ok:
                ok_for_provider = True
                results.append(
                    {
                        "provider": provider,
                        "model": model,
                        "status": status,
                        "latency_sec": latency,
                    }
                )
                if args.stop_on_success:
                    break
            else:
                results.append({"provider": provider, "model": model, "status": status})

            if status == 429 and args.retry_429 > 0:
                time.sleep(args.retry_429)

        if not ok_for_provider:
            fail_count += 1
        if pause > 0:
            time.sleep(pause)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"smoke finished: total_results={len(results)} fail_providers={fail_count}")
        for item in results[:300]:
            print(item)
    return 0 if fail_count == 0 else 3


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model Hotel provider CLI")
    parser.add_argument("--base-api", default="http://192.168.50.130:8081", help="Model Hotel base API")
    parser.add_argument("--admin-token", default=os.getenv("ADMIN_TOKEN"), help="Admin token (or set ADMIN_TOKEN)")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")

    sub = parser.add_subparsers(dest="command", required=True)

    import_cmd = sub.add_parser("import", help="Import providers from key list")
    import_cmd.add_argument("--file", help="Input file path; defaults stdin")
    import_cmd.add_argument("--sep", default="---", help="Field separator")
    import_cmd.add_argument("--strict", action="store_true", help="Stop on malformed lines")
    import_cmd.add_argument("--dry-run", action="store_true", help="Only preview payloads, do not send requests")
    import_cmd.add_argument("--dedup", action="store_true", help="Deduplicate input by account")
    import_cmd.add_argument("--suffix", default="-nvapi", help="Provider name suffix")
    import_cmd.add_argument("--mode", choices=["nvidia", "openrouter"], default="nvidia", help="Preset base URL")
    import_cmd.add_argument("--base-url", default=None, help="Custom provider base_url (overrides --mode)")
    import_cmd.add_argument("--disabled", action="store_true", help="Create provider as disabled")
    import_cmd.add_argument("--no-autodiscovery", action="store_true", help="Disable autodiscovery flag")
    import_cmd.add_argument(
        "--dup-policy",
        default="skip",
        choices=["skip", "fail", "overwrite"],
        help="Conflict handling when same provider exists",
    )
    import_cmd.add_argument("--sleep-ms", type=int, default=0, help="Sleep between create calls")
    import_cmd.add_argument("--extra-payload", help="JSON file merged into payload")
    import_cmd.add_argument("--json", action="store_true", help="Output JSON report")

    discover_cmd = sub.add_parser("discover", help="Run provider discovery (discover-all)")
    discover_cmd.add_argument("--json", action="store_true", help="Output JSON")

    smoke_cmd = sub.add_parser("smoke", help="Run quick completion tests")
    smoke_cmd.add_argument("--providers", nargs="*", help="Target provider names, default all")
    smoke_cmd.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model ids")
    smoke_cmd.add_argument("--prompt", default="1+2=?", help="Prompt for smoke test")
    smoke_cmd.add_argument("--pause-ms", type=int, default=0, help="Pause ms between providers")
    smoke_cmd.add_argument("--stop-on-success", action="store_true", help="Stop on first successful model per provider")
    smoke_cmd.add_argument("--retry-429", type=float, default=0, help="Sleep seconds when 429 returned")
    smoke_cmd.add_argument("--json", action="store_true", help="Output JSON report")

    return parser


def apply_import_defaults(args: argparse.Namespace) -> None:
    if args.command != "import":
        return
    if args.base_url is None:
        args.base_url = PROVIDER_PRESET_BASE[args.mode]
    if args.dup_policy == "overwrite":
        print(
            "overwrite policy selected: attempt duplicates; if server returns errors, "
            "use --extra-payload with provider update payload as needed.",
            file=sys.stderr,
        )


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    apply_import_defaults(args)

    if not args.admin_token:
        parser.error("Admin token is required. Use --admin-token or set ADMIN_TOKEN")

    client = ModelHotelClient(base_api=args.base_api, admin_token=args.admin_token, timeout=args.timeout)

    if args.command == "import":
        return cmd_import(args, client)
    if args.command == "discover":
        return cmd_discover(args, client)
    if args.command == "smoke":
        return cmd_smoke(args, client)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
