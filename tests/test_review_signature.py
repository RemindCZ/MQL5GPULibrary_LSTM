import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from sign_reviews import REVIEW_BOT, SIGNATURE, pull_numbers, sign_pull_request, signed_body


class FakeAPI:
    def __init__(self, author=REVIEW_BOT, kind="Bot", state="COMMENTED"):
        self.item = {"id": 9, "user": {"login": author, "type": kind},
                     "body": "Original review", "state": state}
        self.writes = []
        self.latest = None

    def pages(self, path):
        return [copy.deepcopy(self.item)] if path.endswith("/reviews") else []

    def request(self, path, method="GET", body=None):
        if method == "GET":
            return copy.deepcopy(self.latest or self.item)
        self.writes.append((path, method, body))
        self.item["body"] = body
        return copy.deepcopy(self.item)


class SignatureTests(unittest.TestCase):
    def test_preserves_content_and_appends_exact_signature(self):
        body = "Finding\n\n```cpp\nreturn 1;\n```"
        self.assertEqual(signed_body(body), body + "\n\n---\n\n" + SIGNATURE)

    def test_empty_review_and_idempotence(self):
        self.assertEqual(signed_body(None), SIGNATURE)
        self.assertIsNone(signed_body(SIGNATURE))
        self.assertIsNone(signed_body("Finding\n\n" + SIGNATURE + "\n"))

    def test_only_configured_bot_is_modified(self):
        for author, kind in [("RemindCZ", "User"), ("other[bot]", "Bot"), (REVIEW_BOT, "User")]:
            api = FakeAPI(author, kind)
            self.assertEqual(sign_pull_request(api, 56), 0)
            self.assertFalse(api.writes)

    def test_pending_review_is_not_submitted_or_edited(self):
        api = FakeAPI(state="PENDING")
        self.assertEqual(sign_pull_request(api, 56), 0)
        self.assertFalse(api.writes)

    def test_live_body_is_used_and_second_run_is_noop(self):
        api = FakeAPI()
        api.latest = dict(api.item, body="Newer finding")
        self.assertEqual(sign_pull_request(api, 56), 1)
        self.assertTrue(api.writes[0][2].startswith("Newer finding"))
        api.latest = None
        self.assertEqual(sign_pull_request(api, 56), 0)
        self.assertEqual(len(api.writes), 1)

    def test_event_target_selection(self):
        self.assertEqual(pull_numbers({"pull_request": {"number": 56}}, None, ""), [56])
        self.assertEqual(pull_numbers({"issue": {"number": 56, "pull_request": {}}}, None, ""), [56])
        self.assertEqual(pull_numbers({"issue": {"number": 56, "pull_request": {}}}, None, "57"), [57])

    def test_inline_and_conversation_comments_use_correct_endpoints(self):
        for collection, expected in [("/pulls/56/comments", "/pulls/comments/9"),
                                     ("/issues/56/comments", "/issues/comments/9")]:
            api = FakeAPI()
            api.pages = lambda path: [copy.deepcopy(api.item)] if path == collection else []
            self.assertEqual(sign_pull_request(api, 56), 1)
            self.assertEqual(api.writes[0][:2], (expected, "PATCH"))

    def test_signature_quoted_in_a_finding_is_not_a_final_signature(self):
        body = SIGNATURE + "\nThis is a quoted line in a finding."
        self.assertTrue(signed_body(body).endswith("\n" + SIGNATURE))

    def test_rejects_invalid_request_identifiers(self):
        for number in [0, -1, True, "56"]:
            with self.assertRaises(ValueError):
                sign_pull_request(FakeAPI(), number)


if __name__ == "__main__":
    unittest.main()
