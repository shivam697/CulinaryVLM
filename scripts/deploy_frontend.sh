#!/bin/bash
# ═══════════════════════════════════════════════════════
# CulinaryVLM — Deploy Frontend to GitHub Pages
# ═══════════════════════════════════════════════════════
#
# Usage:
#   ./scripts/deploy_frontend.sh [github-repo-url]
#
# Prerequisites:
#   - Node.js and npm installed
#   - Git configured with GitHub credentials
#   - gh-pages npm package (installed automatically)
#
# Example:
#   ./scripts/deploy_frontend.sh https://github.com/username/culinary-vlm.git

set -e

REPO_URL="${1:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${PROJECT_DIR}/frontend"

echo "═══════════════════════════════════════════════════"
echo "  CulinaryVLM — Deploy Frontend to GitHub Pages"
echo "═══════════════════════════════════════════════════"

cd "$FRONTEND_DIR"

# Check if dist exists or build it
if [ ! -d "dist" ]; then
    echo "Building frontend..."
    npm install
    npm run build
fi

if [ ! -d "dist" ]; then
    echo "ERROR: Build failed — dist/ not created"
    exit 1
fi

echo "Frontend build size: $(du -sh dist/ | cut -f1)"

# Deploy using gh-pages
npm install --save-dev gh-pages 2>/dev/null

if [ -n "$REPO_URL" ]; then
    echo "Deploying to: $REPO_URL"
    npx gh-pages -d dist -r "$REPO_URL" -m "Deploy CulinaryVLM frontend $(date +%Y-%m-%d)"
else
    echo "Deploying to origin..."
    npx gh-pages -d dist -m "Deploy CulinaryVLM frontend $(date +%Y-%m-%d)"
fi

echo ""
echo "✓ Frontend deployed to GitHub Pages!"
echo "  URL: https://<username>.github.io/culinary-vlm/"
