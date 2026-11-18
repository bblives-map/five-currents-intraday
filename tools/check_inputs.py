"""Report private replay prerequisites without connecting to a provider or database."""
import json
import os
from pathlib import Path

root=Path(__file__).resolve().parents[1]
workspace=Path(os.environ.get("FIVE_CURRENTS_DATA_ROOT",str(root)))
manifest=json.loads((root/"private_input_manifest.json").read_text())
missing=[]
for entry in manifest["files"]:
    available=(workspace/entry["path"]).is_file()
    print(("FOUND " if available else "MISSING ")+entry["path"])
    if not available:missing.append(entry["path"])
print(json.dumps({"book":manifest["book"],"missing_private_inputs":len(missing),
                  "providers_contacted":False,"raw_data_distributed":False},indent=2))
