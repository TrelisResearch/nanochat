bash -lc '
set -euo pipefail

# ── 0 · Env toggles ──────────────────────────────────────────────────────────
export PIP_ROOT_USER_ACTION=ignore
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

# ── 1 · System packages ─────────────────────────────────────────────────────
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git nano ninja-build build-essential python3-dev python3-venv \
  libnuma1 libnuma-dev pkg-config curl ca-certificates

# ── 2 · Git identity (optional) ─────────────────────────────────────────────
[ -n "${GIT_USER_NAME:-}"  ] && git config --global user.name  "$GIT_USER_NAME"
[ -n "${GIT_USER_EMAIL:-}" ] && git config --global user.email "$GIT_USER_EMAIL"

# ── 3 · Repo ────────────────────────────────────────────────────────────────
cd /workspace
NANOCHAT_BRANCH="${NANOCHAT_BRANCH:-master}"
if [ -d nanochat/.git ]; then
  git -C nanochat fetch --all
  git -C nanochat checkout "$NANOCHAT_BRANCH"
  git -C nanochat pull --ff-only
else
  if [ -n "${GITHUB_PAT:-}" ]; then
    git clone "https://${GITHUB_PAT}@github.com/TrelisResearch/nanochat.git"
  else
    git clone https://github.com/TrelisResearch/nanochat.git
  fi
  git -C nanochat checkout "$NANOCHAT_BRANCH"
fi
cd nanochat

exec /start.sh
'