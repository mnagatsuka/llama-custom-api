# 🚀 Serena MCP Setup Guide

This guide explains how to install and run the **Serena MCP Server** in a monorepo and connect it with **Claude Code**.

---

## 1. Install `uv` locally (no global install)

```bash
mkdir -p .tools
curl -LsSf https://astral.sh/uv/install.sh | sh
cp ~/.local/bin/uv .tools/
cp ~/.local/bin/uvx .tools/
````

Verify:

```bash
./.tools/uvx --version
```

---

## 2. Create a startup script

Create `scripts/serena.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use repo-local uv and caches
export PATH="$ROOT_DIR/.tools:$PATH"
export UV_CACHE_DIR="$ROOT_DIR/.cache/uv"

exec "$ROOT_DIR/.tools/uvx" \
  --from "git+https://github.com/oraios/serena" \
  serena-mcp-server \
  --context ide-assistant \
  --project "$ROOT_DIR"
```

Make it executable:

```bash
chmod +x scripts/serena.sh
```

Run manually to confirm:

```bash
./scripts/serena.sh
```

You should see Serena starting without errors.

---

## 3. Configure `.mcp.json`

In the repo root, add:

```json
{
  "servers": {
    "serena": {
      "command": "./scripts/serena.sh",
      "cwd": "."
    }
  }
}
```

Claude Code will read this file and automatically spawn Serena.


---

## 4. Configure permissions

Add `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": ["mcp__serena"],
    "deny": []
  }
}
```

---

## 5. Test the connection

1. Restart VS Code (or run **Developer: Reload Window**).
2. Open Claude panel and type `/mcp`.
   → You should see `serena` listed.
3. If not, check logs: **Command Palette → “Claude: Show Logs”**.

   * `ENOENT` = path error → fix `command` in `.mcp.json`
   * `EACCES` = permission error → run `chmod +x scripts/serena.sh`

---

## 6. Optional: manual indexing

You can trigger a one-time full index build:

```bash
./.tools/uvx --from git+https://github.com/oraios/serena serena project index
```

This stores symbols in `.serena/cache/…`.
After that, the MCP server (`scripts/serena.sh`) will keep the index updated automatically.

---

## 7. Ignore generated files

Update `.gitignore`:

```
.tools/
.cache/
.serena/
```

---

## 8. Troubleshooting quick table

| Symptom                   | Fix                                                                             |
| ------------------------- | ------------------------------------------------------------------------------- |
| `✔ connected` never shows | Check `.mcp.json` path → try absolute path to `scripts/serena.sh`               |
| `permission denied`       | `chmod +x scripts/serena.sh` and `.tools/uvx`                                   |
| Multiple servers conflict | Stop manual `serena.sh` before letting VS Code spawn                            |
| Still failing             | Run `./scripts/serena.sh` manually → if it works, problem is `.mcp.json` config |

---

## ✅ Summary

* Use **repo-local uv** in `.tools/`.
* Centralize startup logic in `scripts/serena.sh`.
* Let Claude auto-spawn it via `.mcp.json`.
* Allow access in `.claude/settings.local.json`.
* Run manual `serena project index` once if needed.

With this setup, Claude Code will automatically connect to Serena when you open the repo.
