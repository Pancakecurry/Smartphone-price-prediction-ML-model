"""
patch_mlflow.py — Docker build-time MLflow path rewriter.

NOTE: backend_api.py now bypasses the SQLite tracking store entirely and
loads models via direct filesystem glob — so this script is a belt-and-
suspenders safety net for any residual path references in YAML/text files.

Run once during `docker build`, after `COPY . .`:
    RUN python patch_mlflow.py
"""

import os

# The exact paths crashing the system
MAC_PATH          = "/Users/arnavuppal/8th sem project/smartphone_price_prediction"
MAC_URI_ENCODED   = "file:///Users/arnavuppal/8th%20sem%20project/smartphone_price_prediction"
MAC_URI_PLAIN     = "file:///Users/arnavuppal/8th sem project/smartphone_price_prediction"

NEW_PATH = "/app"
NEW_URI  = "file:///app"

print("Initiating Brute-Force MLflow Path Patcher...")
patched_files = 0

for root, dirs, files in os.walk("mlruns"):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if MAC_PATH in content or MAC_URI_ENCODED in content or MAC_URI_PLAIN in content:
                # Execute brute-force replacements — order matters (encoded first)
                content = content.replace(MAC_URI_ENCODED, NEW_URI)
                content = content.replace(MAC_URI_PLAIN,   NEW_URI)
                content = content.replace(MAC_PATH,         NEW_PATH)

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                patched_files += 1

        except Exception:
            # Ignore binary files (model.pkl, .pth, sqlite, etc.)
            pass

print(f"Patch complete. Successfully rewritten paths in {patched_files} files.")
