#!/usr/bin/env bash
# Copy the live estimator's logs into the checkout and push them to the running branch
# (used by live-chunk.yml). The working tree only changes here, so pull --rebase is safe.
set -u
cp -r "$LIVE_LOG_DIR"/. logs/
git add -f logs/live_summary.json logs/live_predictions.csv logs/live_stdout.txt logs/quotes 2>/dev/null
git diff --cached --quiet && exit 0
git commit -qm "live logs $(date -u +%Y-%m-%dT%H:%MZ)" || exit 0
for i in 1 2 3; do
  git pull -q --rebase origin "$GITHUB_REF_NAME" && git push -q origin "HEAD:$GITHUB_REF_NAME" && exit 0
  sleep $((5 * i))
done
echo "log push failed" >&2
