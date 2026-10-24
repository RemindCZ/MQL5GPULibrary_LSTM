# Review signature

Review summaries and inline comments use the text signature:

> TBR Tomáš Bělák Remind

`AGENTS.md` requests the signature in generated review text. The **Review
signature** workflow also appends it to unsigned review summaries, inline
comments, and pull-request conversation comments from
`chatgpt-codex-connector[bot]`. GitHub continues to show the original bot author;
the footer does not assert personal review or approval by a human.

The workflow edits existing messages and preserves their contents and review
state. Human comments and other bots are excluded. Repeated runs do not duplicate
the signature. The library and its runtime behavior are unaffected.

Signing happens after publication, so a message can briefly appear unsigned.
Events normally trigger signing immediately. A scan of open pull requests runs
every 15 minutes as a fallback for missed events and fork pull requests, whose
review events have read-only tokens. GitHub can delay scheduled runs. Closed
pull requests can be processed using **Actions → Review signature → Run workflow**
and the pull-request number; an empty number scans all open pull requests.

The workflow uses GitHub's temporary token with `contents: read` and
`pull-requests: write`. No personal token or extra secret is needed. It always
loads the signing code from the repository's default branch and does not execute
pull-request code. If an API update fails, the workflow fails visibly instead of
reporting success. Tests run before any message is updated:

```shell
python tests/test_review_signature.py
```

This workflow does not change the provider's automatic-review settings.
