import base64
import json
import unittest

from omega16_tpu_colab import create_app, decode_genome_payload


class Omega16DecodeTests(unittest.TestCase):
    def test_decode_genome_payload_valid(self):
        payload = {"version": "L41", "status": "OK"}
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
        decoded = decode_genome_payload(encoded)
        self.assertEqual(decoded["version"], "L41")
        self.assertEqual(decoded["status"], "OK")

    def test_decode_genome_payload_invalid(self):
        decoded = decode_genome_payload("not-base64")
        self.assertEqual(decoded["decode_status"], "failed")


class Omega16AuditApiTests(unittest.TestCase):
    def setUp(self):
        try:
            import flask  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("Flask is not installed in this environment.")
        self.app = create_app()
        self.client = self.app.test_client()

    def test_audit_sync_and_state(self):
        payload = {
            "version": "L41_RECURSIVE_TRANSCENDENT_SELF",
            "timestamp": "2026-04-05T14:04:48.403710",
            "modules": {"01/CORE/IDENTITY_CELL.txt": {"length": 81}},
        }
        r = self.client.post("/audit/sync", json=payload)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["synced_version"], payload["version"])

        state = self.client.get("/audit/state")
        self.assertEqual(state.status_code, 200)
        sbody = state.get_json()
        self.assertTrue(sbody["runtime_synced"])
        self.assertEqual(sbody["runtime_payload_version"], payload["version"])


if __name__ == "__main__":
    unittest.main()
