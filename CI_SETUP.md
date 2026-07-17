# Setting up CI on the GitHub repo

GitHub protects workflow files (`/.github/workflows/*.yml`) by requiring
an OAuth app to have the `workflow` scope before it can push them.
The `gh` CLI we used to create the repo doesn't have that scope by
default, so the workflow file is sitting on a local commit that
couldn't be pushed.

## How to enable CI

**Easiest: do it in the web UI.**

1. Go to https://github.com/kajica2/solokit/settings/actions
2. Under "Workflow permissions", select "Read and write permissions"
   and check "Allow GitHub Actions to create and approve pull requests".
3. Click "Save".

Then add the workflow file:

1. Go to https://github.com/kajica2/solokit/tree/master/.github
2. Click "Add file" → "Create new file"
3. Name: `workflows/ci.yml`
4. Paste the contents of `.github/workflows/ci.yml` from the local
   checkout (the file exists at
   `/Users/kajicadjuric/Documents/research/solokit/.github/workflows/ci.yml`
   after running the commands below)
5. Commit directly to master

**Alternative: re-authorize the gh CLI with workflow scope.**

```bash
gh auth refresh -h github.com -s workflow
# This opens a browser to re-authorize. After completing, retry the push:
cd /Users/kajicadjuric/Documents/research/solokit
git checkout 09b872a -- .github/workflows/ci.yml
git commit -m "Restore CI workflow"
git push origin master
```

**Alternative: push from your own git config (not via gh CLI).**

If you have a personal access token (PAT) with `workflow` scope, you
can push with that instead of the gh CLI's limited OAuth token:

```bash
# in the GitHub web UI: Settings → Developer settings → Personal access
# tokens → Tokens (classic) → Generate new token
# Scopes: repo, workflow
export GITHUB_TOKEN=ghp_xxxxxxxx
git remote set-url origin https://x-access-token:$GITHUB_TOKEN@github.com/kajica2/solokit.git
git push origin master
git remote set-url origin https://github.com/kajica2/solokit.git  # restore
unset GITHUB_TOKEN
```

## What the CI does

`.github/workflows/ci.yml` runs on push and PR:
- Tests on Python 3.11 and 3.12
- Installs `solokit[dev]` (no audio deps — too heavy for CI)
- Runs `pytest tests/ -v --tb=short -m "not audio"`
- Runs `ruff check src/ tests/`

It does NOT install the heavy `[audio]` extras (PyTorch ~1GB) and
skips audio-marked tests. E2e tests (Puppeteer) are also excluded —
they need a running uvicorn and browser.

129 tests pass locally in ~13s.
