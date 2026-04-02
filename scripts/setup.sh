#!/bin/bash
# setup.sh - Clone all code repositories and fetch arxiv papers
# Usage: ./scripts/setup.sh [--repos | --all]
#   (default)  fetch arxiv papers only (notes + markdown)
#   --repos    clone code repositories only
#   --all      fetch papers + clone repos
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MANIFEST="$ROOT_DIR/papers.yaml"

cd "$ROOT_DIR"

# ============================================================
# Parse papers.yaml and clone repos
# Requires: python3 + pyyaml
# ============================================================
clone_repos() {
    echo "=== Cloning code repositories ==="
    python3 - "$MANIFEST" <<'PYEOF'
import yaml, subprocess, sys, os

with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)

root = os.path.dirname(sys.argv[1])
total, cloned, skipped = 0, 0, 0

for section in data.values():
    if not isinstance(section, list):
        continue
    for paper in section:
        folder = paper.get("folder", "")
        for repo in paper.get("repos", []):
            total += 1
            name = repo["name"]
            url = repo["url"]
            if name == ".":
                target = os.path.join(root, folder)
            else:
                target = os.path.join(root, folder, name)

            if os.path.isdir(os.path.join(target, ".git")):
                print(f"  [skip] {folder}/{name} (already cloned)")
                skipped += 1
                continue

            os.makedirs(os.path.dirname(target), exist_ok=True)
            print(f"  [clone] {url} -> {folder}/{name}")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", url, target],
                    check=True, capture_output=True, text=True,
                )
                cloned += 1
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] Failed to clone {url}: {e.stderr.strip()}")

print(f"\nDone: {cloned} cloned, {skipped} skipped, {total} total repos")
PYEOF
}

# ============================================================
# Fetch arxiv papers as markdown (where HTML is available)
# ============================================================
fetch_papers() {
    echo "=== Fetching arxiv papers ==="
    ARXIV2MD="$ROOT_DIR/html2aitext_convert/arxiv2md.sh"
    if [ ! -f "$ARXIV2MD" ]; then
        echo "Warning: arxiv2md.sh not found, skipping paper fetch"
        echo "Clone html2aitext_convert first: git clone <url> html2aitext_convert"
        return
    fi

    python3 - "$MANIFEST" "$ARXIV2MD" "$ROOT_DIR" <<'PYEOF'
import yaml, subprocess, sys, os

with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)

arxiv2md = sys.argv[2]
root = sys.argv[3]
fetched, skipped, failed = 0, 0, 0

for section in data.values():
    if not isinstance(section, list):
        continue
    for paper in section:
        folder = paper.get("folder", "")
        target_dir = os.path.join(root, folder)

        # Check for arxiv ID
        arxiv_id = paper.get("arxiv")
        if not arxiv_id:
            continue

        # Check if markdown already exists
        md_files = [f for f in os.listdir(target_dir) if f.endswith(".md") and f != "CLAUDE.md" and not f.endswith("_notes.md")]
        if md_files:
            print(f"  [skip] {folder} (markdown exists: {md_files[0]})")
            skipped += 1
            continue

        print(f"  [fetch] arxiv:{arxiv_id} -> {folder}")
        try:
            result = subprocess.run(
                ["bash", arxiv2md, arxiv_id],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                # Find the output file and copy it
                for line in result.stdout.split("\n"):
                    if "__OUTPUT_FILE__:" in line:
                        src = line.split(":", 1)[1].strip()
                        if os.path.exists(src):
                            dest = os.path.join(target_dir, os.path.basename(src))
                            os.makedirs(target_dir, exist_ok=True)
                            import shutil
                            shutil.copy2(src, dest)
                            fetched += 1
                            break
            else:
                # Fallback: download PDF
                print(f"    [pdf] HTML not available, downloading PDF...")
                title = paper.get("title", arxiv_id).replace(" ", "_").replace(":", "")[:80]
                pdf_path = os.path.join(target_dir, f"{title}.pdf")
                if not os.path.exists(pdf_path):
                    subprocess.run(
                        ["curl", "-sL", f"https://arxiv.org/pdf/{arxiv_id}", "-o", pdf_path],
                        check=True, timeout=60,
                    )
                fetched += 1
        except Exception as e:
            print(f"    [ERROR] {e}")
            failed += 1

print(f"\nDone: {fetched} fetched, {skipped} skipped, {failed} failed")
PYEOF
}

# ============================================================
# Main
# ============================================================
case "${1:-}" in
    --repos)       clone_repos ;;
    --all)         clone_repos; echo; fetch_papers ;;
    *)             fetch_papers ;;
esac
