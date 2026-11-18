"""Fail closed on credential patterns and accidentally staged private data."""
import re
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
listed=subprocess.run(["git","-C",str(root),"ls-files","--cached"],capture_output=True,text=True)
paths=[root/name for name in listed.stdout.splitlines()] if listed.returncode==0 and listed.stdout.strip() else [
    path for path in root.rglob("*") if path.is_file() and not any(part in {".git","__pycache__",".pytest_cache",".venv","build"} or part.endswith(".egg-info") for part in path.relative_to(root).parts)]
forbidden={".parquet",".pkl",".npy",".npz",".db",".sqlite",".pdf",".dbn"}
patterns=[re.compile(r"(?i)(?:postgres(?:ql)?|mysql|redis)://[^\s\"']+:[^\s\"'@/]+@"),
          re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|db-[A-Za-z0-9]{25,}")]
issues=[]
for path in paths:
    if path.suffix.lower() in forbidden or path.name in {"admin_access.json","Open_Admin_Dashboard.html",".env"}:
        issues.append(str(path.relative_to(root))+": private artifact")
    if path.suffix.lower() not in {".png",".jpg",".gif",".woff",".woff2"}:
        try:text=path.read_text(encoding="utf-8")
        except UnicodeError:
            issues.append(str(path.relative_to(root))+": unreviewed binary");continue
        for pattern in patterns:
            if pattern.search(text):issues.append(str(path.relative_to(root))+": credential pattern")
if issues:raise SystemExit("Public release blocked:\n"+"\n".join(issues))
print(f"Public release scan passed: {len(paths)} files; no private artifacts or credential patterns")
