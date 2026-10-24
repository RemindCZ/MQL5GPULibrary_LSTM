"""Append the repository signature to the configured review bot's output."""
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

SIGNATURE = "TBR Tomáš Bělák Remind"
REVIEW_BOT = "chatgpt-codex-connector[bot]"


def signed_body(body):
    text = (body or "").rstrip()
    if text == SIGNATURE or text.endswith("\n" + SIGNATURE):
        return None
    return (text + "\n\n---\n\n" if text else "") + SIGNATURE


def is_review_bot(item):
    user = item.get("user") or {}
    return user.get("login") == REVIEW_BOT and user.get("type") == "Bot"


class GitHub:
    def __init__(self, repository, token):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Invalid repository name")
        self.base = "https://api.github.com/repos/" + repository
        self.token = token

    def request(self, path, method="GET", body=None):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Expected a repository-relative API path")
        data = None if body is None else json.dumps({"body": body}).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data, method=method,
            headers={"Authorization": "Bearer " + self.token,
                     "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28",
                     "Content-Type": "application/json",
                     "User-Agent": "Repository-Review-Signature"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            return json.loads(payload) if payload else None

    def pages(self, path):
        separator = "&" if "?" in path else "?"
        for page in range(1, 101):
            rows = self.request(f"{path}{separator}per_page=100&page={page}")
            yield from rows
            if len(rows) < 100:
                return
        raise RuntimeError("API pagination exceeded the supported limit")


def sign_pull_request(api, number):
    if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
        raise ValueError("Expected a positive pull request number")
    collections = [
        (f"/pulls/{number}/reviews", "PUT", lambda item: f"/pulls/{number}/reviews/{int(item['id'])}"),
        (f"/pulls/{number}/comments", "PATCH", lambda item: f"/pulls/comments/{int(item['id'])}"),
        (f"/issues/{number}/comments", "PATCH", lambda item: f"/issues/comments/{int(item['id'])}"),
    ]
    changed = 0
    for collection, method, location in collections:
        for item in api.pages(collection):
            if not is_review_bot(item) or item.get("state") == "PENDING":
                continue
            path = location(item)
            try:
                # Re-read immediately before editing; webhook bodies may be stale.
                current = api.request(path)
                if not is_review_bot(current) or current.get("state") == "PENDING":
                    continue
                body = signed_body(current.get("body"))
                if body is None:
                    continue
                updated = api.request(path, method, body)
                if updated.get("body") != body:
                    raise RuntimeError("Signature update was not confirmed")
                changed += 1
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
    return changed


def pull_numbers(event, api, manual_number):
    if manual_number:
        number = int(manual_number)
        if number <= 0:
            raise ValueError("Pull request number must be positive")
        return [number]
    if "pull_request" in event:
        return [int(event["pull_request"]["number"])]
    if "pull_request" in event.get("issue", {}):
        return [int(event["issue"]["number"])]
    return [int(pr["number"]) for pr in api.pages("/pulls?state=open")]


def main():
    api = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    numbers = pull_numbers(event, api, os.environ.get("REVIEW_PULL_NUMBER", "").strip())
    changed = sum(sign_pull_request(api, number) for number in numbers)
    print(f"Checked {len(numbers)} pull requests; signed {changed} review messages.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        print(f"GitHub returned HTTP {error.code}; review signing failed.", file=sys.stderr)
        raise SystemExit(1)
