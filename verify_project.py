"""Verify the published contribution against its recorded file inventory."""
from pathlib import Path
import hashlib
import json
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'PROJECT_MANIFEST.json').read_text())
problems = []
for entry in manifest['files']:
    path = root / entry['path']
    if not path.is_file():
        problems.append(f"Missing: {entry['path']}")
    elif path.stat().st_size != entry['size_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        problems.append(f"Changed: {entry['path']}")
if problems:
    print('\n'.join(problems))
    sys.exit(1)
print(f"Verified {len(manifest['files'])} contribution files ({manifest['total_size_bytes']:,} bytes).")
