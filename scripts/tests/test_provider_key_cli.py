#!/usr/bin/env python3
"""Unit tests for provider_key_cli.py."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import provider_key_cli as cli


def _write_tmp_file(content: str) -> str:
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as fp:
        fp.write(content)
        return fp.name


@dataclass
class FakeClient:
    providers: list[dict[str, Any]] = field(default_factory=list)
    create_results: list[tuple[bool, int, Any]] = field(default_factory=list)
    chat_results: list[tuple[bool, int, Any, float]] = field(default_factory=list)
    list_calls: int = 0
    create_calls: list[dict[str, Any]] = field(default_factory=list)
    chat_calls: list[tuple[str, str, str]] = field(default_factory=list)

    def list_providers(self) -> list[dict[str, Any]]:
        self.list_calls += 1
        return self.providers

    def create_provider(self, payload: dict[str, Any]) -> tuple[bool, int, Any]:
        self.create_calls.append(payload)
        if self.create_results:
            return self.create_results.pop(0)
        return True, 201, {"ok": True}

    def chat_completion(self, provider: str, model: str, prompt: str) -> tuple[bool, int, Any, float]:
        self.chat_calls.append((provider, model, prompt))
        if self.chat_results:
            return self.chat_results.pop(0)
        return True, 200, {"ok": True}, 0.0


class TestProviderKeyCli(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = cli.make_parser()

    def test_parse_records_accepts_clean_lines(self) -> None:
        text = (
            " a1 --- p1 --- j1 --- key1 \n"
            "# comment line\n"
            "\n"
            "a2---p2---j2---key2"
        )
        records = list(cli.parse_records(text, "---", strict=False))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].account, "a1")
        self.assertEqual(records[0].password, "p1")
        self.assertEqual(records[1].api_key, "key2")

    def test_parse_records_skip_malformed_in_relaxed_mode(self) -> None:
        text = "a---p\ninvalid---line\nb---p2---j2---k2\n"
        records = list(cli.parse_records(text, "---", strict=False))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].account, "b")

    def test_parse_records_strict_mode_raises_on_invalid(self) -> None:
        text = "a---p\n"
        with self.assertRaises(ValueError):
            list(cli.parse_records(text, "---", strict=True))

    def test_dedupe_keeps_first_occurrence(self) -> None:
        items = [
            cli.KeyRecord(account="acc", password="p1", email_jwt="j1", api_key="k1"),
            cli.KeyRecord(account="acc", password="p2", email_jwt="j2", api_key="k2"),
            cli.KeyRecord(account="acc3", password="p3", email_jwt="j3", api_key="k3"),
        ]
        result = list(cli.dedupe(items))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].password, "p1")
        self.assertEqual(result[1].account, "acc3")

    def test_normalize_provider_name(self) -> None:
        self.assertEqual(cli.normalize_provider_name("foo", "-nvapi"), "foo-nvapi")
        self.assertEqual(cli.normalize_provider_name("foo-nvapi", "-nvapi"), "foo-nvapi")

    def test_build_payload_variants(self) -> None:
        variants = cli.build_payload_variants(
            provider_name="acc-nvapi",
            base_url="https://example.com/v1",
            api_key="k",
            enabled=True,
            autodiscovery=False,
        )
        self.assertEqual(len(variants), 4)
        self.assertTrue(all(v["name"] == "acc-nvapi" for v in variants))
        self.assertTrue(any("provider_type" in v for v in variants))
        self.assertTrue(any("providerType" in v for v in variants))
        self.assertTrue(any("baseUrl" in v for v in variants))

    def test_sanitize_payload_redacts_sensitive_keys(self) -> None:
        payload = {
            "name": "acc",
            "api_key": "secret",
            "nested": {"api_key": "nested-secret"},
            "apiKey": "other-secret",
            "authorization": "bearer",
        }
        sanitized = cli.sanitize_payload(payload)
        self.assertEqual(sanitized["api_key"], "***redacted***")
        self.assertEqual(sanitized["apiKey"], "***redacted***")
        self.assertEqual(sanitized["authorization"], "***redacted***")
        self.assertEqual(sanitized["nested"]["api_key"], "nested-secret")

    def test_read_input_from_file_and_stdin(self) -> None:
        file_path = _write_tmp_file("hello")
        try:
            self.assertEqual(cli.read_input(file_path), "hello")
            with patch("sys.stdin", io.StringIO("stdin-line")):
                self.assertEqual(cli.read_input(None), "stdin-line")
        finally:
            os.unlink(file_path)

    def test_apply_import_defaults_uses_mode_preset(self) -> None:
        args = self.parser.parse_args(["import", "--mode", "openrouter"])
        cli.apply_import_defaults(args)
        self.assertEqual(args.base_url, cli.PROVIDER_PRESET_BASE["openrouter"])

    def test_apply_import_defaults_uses_nas_env_by_default(self) -> None:
        args = self.parser.parse_args(["import"])
        cli.apply_import_defaults(args)
        self.assertEqual(args.base_api, cli.ENV_PRESET_BASE_API["nas"])

    def test_apply_import_defaults_respects_base_api_override(self) -> None:
        custom_api = "http://127.0.0.1:9999"
        args = self.parser.parse_args(["--base-api", custom_api, "--env", "nas", "import"])
        cli.apply_import_defaults(args)
        self.assertEqual(args.base_api, custom_api)

    def _run_cmd_import(self, args: list[str], client: FakeClient) -> tuple[dict[str, Any], int]:
        parsed = self.parser.parse_args(["import", *args])
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            rc = cli.cmd_import(parsed, client)
            output = stdout.getvalue().strip()
        if parsed.json:
            marker = output.find("\n{")
            if marker == -1:
                return json.loads(output), rc
            return json.loads(output[marker + 1 :]), rc
        return {"created": [], "skipped": [], "failed": []}, rc

    def test_cmd_import_dry_run_outputs_payload_preview(self) -> None:
        path = _write_tmp_file("acc---p---j---k\n")
        client = FakeClient()
        try:
            report, rc = self._run_cmd_import(["--file", path, "--json", "--dry-run", "--dedup"], client)
            self.assertEqual(rc, 0)
            self.assertEqual(len(report["created"]), 1)
            self.assertEqual(report["created"][0]["provider"], "acc-nvapi")
            self.assertEqual(report["created"][0]["status"], "dry-run")
            self.assertEqual(report["created"][0]["payload"]["api_key"], "***redacted***")
            self.assertEqual(client.create_calls, [])
        finally:
            os.unlink(path)

    def test_cmd_import_duplicate_policy_skip(self) -> None:
        path = _write_tmp_file("acc---p---j---k\n")
        client = FakeClient(providers=[{"name": "acc-nvapi"}])
        try:
            report, rc = self._run_cmd_import(["--file", path, "--json", "--dup-policy", "skip"], client)
            self.assertEqual(rc, 0)
            self.assertEqual(
                report["skipped"],
                [{"provider": "acc-nvapi", "reason": "already_exists"}],
            )
            self.assertEqual(report["created"], [])
            self.assertEqual(client.create_calls, [])
        finally:
            os.unlink(path)

    def test_cmd_import_duplicate_policy_fail(self) -> None:
        path = _write_tmp_file("acc---p---j---k\n")
        client = FakeClient(providers=[{"name": "acc-nvapi"}])
        try:
            report, rc = self._run_cmd_import(["--file", path, "--json", "--dup-policy", "fail"], client)
            self.assertEqual(rc, 2)
            self.assertEqual(
                report["failed"],
                [{"provider": "acc-nvapi", "reason": "already_exists"}],
            )
            self.assertEqual(client.create_calls, [])
        finally:
            os.unlink(path)

    def test_cmd_import_overwrite_tries_alternate_payloads(self) -> None:
        path = _write_tmp_file("acc---p---j---k\n")
        client = FakeClient(
            create_results=[
                (False, 400, {"error": "validation"}),
                (False, 422, {"error": "schema"}),
                (True, 201, {"ok": True}),
            ]
        )
        try:
            report, rc = self._run_cmd_import(["--file", path, "--json", "--dup-policy", "overwrite"], client)
            self.assertEqual(rc, 0)
            self.assertEqual(len(report["created"]), 1)
            self.assertEqual(len(client.create_calls), 3)
        finally:
            os.unlink(path)

    def _run_smoke(self, args: list[str], client: FakeClient) -> tuple[Any, int, str]:
        parsed = self.parser.parse_args(["smoke", *args])
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            rc = cli.cmd_smoke(parsed, client)
            output = stdout.getvalue().strip()
        if parsed.json and output:
            try:
                return json.loads(output), rc, output
            except json.JSONDecodeError:
                return [], rc, output
        return [], rc, output

    def test_cmd_smoke_no_providers_returns_error(self) -> None:
        client = FakeClient(providers=[])
        report, rc, out = self._run_smoke(["--json"], client)
        self.assertEqual(rc, 1)
        self.assertEqual(report, [])
        self.assertIn("No providers available", out)

    def test_cmd_smoke_unknown_provider_is_failure(self) -> None:
        client = FakeClient(providers=[{"name": "acc1"}])
        report, rc, _out = self._run_smoke(["--json", "--providers", "not_exist"], client)
        self.assertEqual(rc, 3)
        self.assertEqual(report, [{"provider": "not_exist", "status": "not_found"}])
        self.assertEqual(client.chat_calls, [])

    def test_cmd_smoke_success_path_returns_zero(self) -> None:
        client = FakeClient(
            providers=[{"name": "acc1"}],
            chat_results=[(True, 200, {"ok": True}, 0.12)],
        )
        report, rc, _out = self._run_smoke(["--json", "--providers", "acc1", "--models", "m1"], client)
        self.assertEqual(rc, 0)
        self.assertEqual(
            report,
            [{"provider": "acc1", "model": "m1", "status": 200, "latency_sec": 0.12}],
        )

    def test_cmd_smoke_stop_on_success_stops_after_first_success(self) -> None:
        client = FakeClient(
            providers=[{"name": "acc1"}],
            chat_results=[
                (True, 200, {"ok": True}, 0.01),
                (False, 500, {"err": "later"}, 0.02),
                (False, 500, {"err": "later2"}, 0.03),
            ],
        )
        report, rc, _out = self._run_smoke(
            ["--json", "--providers", "acc1", "--models", "m1,m2,m3", "--stop-on-success"],
            client,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["status"], 200)


if __name__ == "__main__":
    unittest.main()
