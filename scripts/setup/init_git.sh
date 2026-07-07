#!/usr/bin/env bash
# One-time git bootstrap for the DC-ASR project.
# Run this ON YOUR MAC from the project root (the agent sandbox cannot create .git):
#     bash scripts/init_git.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# --- commit identity (edit if you want something other than your GitHub identity) ---
GIT_NAME="${GIT_NAME:-Anshul Kumar}"
GIT_EMAIL="${GIT_EMAIL:-anshulk-cmu@users.noreply.github.com}"   # <- set to your real git email

git init -b main
git config --local user.name  "$GIT_NAME"
git config --local user.email "$GIT_EMAIL"

git add -A
git commit -m "Initial commit: DC-ASR project scaffold

- docs/: experimental plan (v3) + research proposal
- src/dcasr/: package stubs (data, models[mamba/hnet/encoder], decoders, training, interp)
- configs/typeA_small_N1_ctc.yaml: first go/no-go experiment (Type A, Small, N=1, CTC)
- README, requirements.txt, .gitignore (data/ckpt/logs excluded)"

echo
echo "Done. Local repo initialized on branch 'main' with one commit."
echo "To push to GitHub, create an empty repo (no README) then:"
echo "    git remote add origin git@github.com:anshulk-cmu/<repo-name>.git"
echo "    git push -u origin main"
