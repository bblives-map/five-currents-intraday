"""Create source-only, credential-sanitized research exports; never run collectors."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))
OUTPUT = ROOT / "public_repositories"
INTERDAY_TESTS = {
    "test_five_currents_daily.py", "test_five_currents_dashboard.py",
    "test_five_currents_reconciliation.py", "test_five_currents_mixed_curve.py",
    "test_five_currents_provisional_curve.py", "test_five_currents_shadow_execution.py",
    "test_five_currents_shadow_bootstrap.py", "test_five_currents_source_admission.py",
    "test_five_currents_oi_release.py", "test_five_currents_cme_failsafe.py",
    "test_five_currents_cme_reference.py", "test_five_currents_databento_failsafe.py",
    "test_five_currents_live_mtm.py", "test_five_currents_contract_retention.py",
    "test_five_currents_daily_reconciliation_audit.py",
}
DAILY_ENTRIES = {
    "run_five_currents_daily.py", "run_five_currents_dashboard.py",
    "activate_five_currents_mixed_curve.py", "initialize_five_currents_shadow.py",
    "repair_five_currents_roll_marks.py", "verify_five_currents_dashboard.py",
    "five_currents_daily_reconciliation_audit.py", "probe_five_currents_cme.py",
    "probe_five_currents_databento.py",
}
URI = re.compile(r"(?i)(?:postgres(?:ql)?|mysql|redis)://[^\s\"'<>]+")
SECRET_NAMES = re.compile(r"(?i)(?:password|passwd|api_?key|access_?token|auth_?token|client_?secret)$")
USER_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\r\n]+(?:[\\/]+[^\r\n\"'<>]*)?")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_path(name):
    base = ROOT.joinpath(*name.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path.relative_to(ROOT)
    return None


def imports(path, text):
    package = list(path.parts[:-1])
    names = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package[:len(package) - node.level + 1] if node.level else []
            base = ".".join([*prefix, *([node.module] if node.module else [])])
            if base:
                names.add(base)
                names.update(base + "." + alias.name for alias in node.names if alias.name != "*")
    return names


def export_text(path, text, book):
    """Preserve executable logic; remove only credentials and personal paths."""
    findings = []
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offsets, offset = [], 0
    for line in lines:
        offsets.append(offset); offset += len(line)
    replacements = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        clean = value
        if URI.search(clean):
            def safe_uri(match):
                value = urlsplit(match.group(0))
                host = value.hostname or "localhost"
                return urlunsplit((value.scheme, host + (f":{value.port}" if value.port else ""), value.path, value.query, value.fragment))
            clean = URI.sub(safe_uri, clean)
        if re.match(r"(?i)^[a-z]:[\\/]Users[\\/]", clean):
            marker = re.search(r"(?i)[\\/]jod[\\/]", clean)
            clean = clean[marker.end():].replace("\\", "/") if marker else "private_inputs/" + Path(clean.replace("\\", "/")).name
        if clean != value:
            findings.append({"file": str(path), "line": node.lineno, "kind": "private connection/path replaced"})
            begin = offsets[node.lineno-1] + node.col_offset
            finish = offsets[node.end_lineno-1] + node.end_col_offset
            replacements.append((begin, finish, repr(clean)))
    for begin, finish, replacement in sorted(replacements, reverse=True):
        text = text[:begin] + replacement + text[finish:]
    # Preserve source layout while removing private owner-path comments/docstrings.
    text = re.sub(r"(?i)[A-Z]:[\\/]+Users[\\/]+Janardan Amit[\\/]+jod[\\/]+", "", text)
    text = text.replace("from five_currents_optional_bloomberg import Terminal", "from five_currents_optional_bloomberg import Terminal")
    if path.name == "five_currents_daily_reconciliation_audit.py":
        text = text.replace("from five_currents_intraday_replay import ROOTS", "from us_commodity_research_core import PRODUCT_ORDER as ROOTS")
    if path.name == "test_five_currents_contract_retention.py":
        text = text.replace('"maintenance/roll_safe_daily_20261028/ibkr_daily_curve_jobs.py"', '"ibkr_daily_curve_jobs.py"')
    if path.as_posix() == "five_currents_daily/dashboard_worker.py" and book == "interday":
        text = text.replace(
            "from calculate_five_currents_intraday_positions import SOURCE as INTRADAY_SOURCE, build_exposure",
            "import importlib\ntry:\n    _intraday = importlib.import_module('calculate_five_currents_intraday_positions')\n"
            "    INTRADAY_SOURCE, build_exposure = _intraday.SOURCE, _intraday.build_exposure\n"
            "except ModuleNotFoundError as exc:\n    if exc.name != 'calculate_five_currents_intraday_positions':\n        raise\n"
            "    INTRADAY_SOURCE, build_exposure = None, None")
        text = text.replace("    now = now or datetime.now(timezone.utc)\n    preview = public.get(\"provisional\")",
            "    now = now or datetime.now(timezone.utc)\n    if build_exposure is None:\n"
            "        schedule = {'ready': False, 'state': 'not_installed', 'signal_date': public['signal_date'],\n"
            "                    'status': 'Install the five-currents-intraday companion for intraday exposure'}\n"
            "        public['intraday_exposure'] = admin['intraday_exposure'] = schedule\n        return\n"
            "    preview = public.get(\"provisional\")")
        text = text.replace("        digest.update((directory / name).read_bytes())",
                            "        if (directory / name).is_file():\n            digest.update((directory / name).read_bytes())")
        text = text.replace("    if include_output:\n        winners = INTRADAY_SOURCE", "    if include_output and INTRADAY_SOURCE is not None:\n        winners = INTRADAY_SOURCE")
        text = text.replace("        digest.update(winners.read_bytes())", "        if winners.is_file():\n            digest.update(winners.read_bytes())")
    # A package install must use an explicit shared private workspace, not site-packages.
    text = re.sub(r"(?m)^ROOT = Path\(__file__\)\.resolve\(\)\.parent\s*$",
                  "ROOT = Path(__import__('os').environ.get('FIVE_CURRENTS_DATA_ROOT', str(Path(__file__).resolve().parent)))", text)
    if len(path.parts)==1:
        text = re.sub(r'Path\(__file__\)\.resolve\(\)\.parent / "(?=(?:five_currents_|us_commodity_))',
            'Path(__import__("os").environ.get("FIVE_CURRENTS_DATA_ROOT", str(Path(__file__).resolve().parent))) / "',text)
    ast.parse(text)
    return text, findings


def seeds(book):
    if book == "interday":
        paths = {path.relative_to(ROOT) for pattern in ("run_us_commodity*.py", "us_commodity*core.py", "test_us_commodity*.py")
                 for path in ROOT.glob(pattern)}
        paths |= {Path(name) for name in DAILY_ENTRIES | INTERDAY_TESTS if (ROOT/name).is_file()}
        paths |= {path.relative_to(ROOT) for path in (ROOT/"five_currents_daily").glob("*.py")}
        paths.add(Path("build_us_commodity_five_currents_pitch.py"))
    else:
        paths = {path.relative_to(ROOT) for pattern in ("*five_currents*.py", "test_reconcile_five_currents*.py")
                 for path in ROOT.glob(pattern)}
        paths = {path for path in paths if path.name not in DAILY_ENTRIES | INTERDAY_TESTS}
        paths = {path for path in paths if not path.name.startswith(("inspect_", "profile_", "refresh_", "repair_"))}
    return paths


def auxiliary(path):
    return path.name.startswith(("shfe_", "run_shfe_")) or path.name in {
        "energy_equity_oil_leadlag_study.py", "bbg_research_pull_to_postgres.py",
        "pull_master_research_daily_fields.py"}


def slim_shared_module(text, requested):
    tree=ast.parse(text)
    definitions={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            definitions[node.name]=node
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for target in targets:
                for item in ast.walk(target):
                    if isinstance(item,ast.Name):definitions[item.id]=node
    required=set(requested); pending=list(required)
    while pending:
        name=pending.pop(); node=definitions.get(name)
        if node is None:continue
        loads={item.id for item in ast.walk(node) if isinstance(item,ast.Name) and isinstance(item.ctx,ast.Load)}
        for dep in loads-required:
            required.add(dep)
            if dep in definitions:pending.append(dep)
    kept=[]
    needed_nodes={id(definitions[name]) for name in required if name in definitions}
    for node in tree.body:
        if id(node) in needed_nodes:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.decorator_list:
                source_lines=text.splitlines()
                kept.append("\n".join(source_lines[node.decorator_list[0].lineno-1:node.end_lineno]))
            else:kept.append(ast.get_source_segment(text,node))
        elif isinstance(node,ast.ImportFrom):
            names=[alias for alias in node.names if (alias.asname or alias.name) in required or node.module=="__future__"]
            if names:
                prefix="."*node.level+(node.module or "")
                kept.append("from "+prefix+" import "+", ".join(alias.name+(" as "+alias.asname if alias.asname else "") for alias in names))
        elif isinstance(node,ast.Import):
            names=[alias for alias in node.names if (alias.asname or alias.name.split('.')[0]) in required]
            if names:kept.append("import "+", ".join(alias.name+(" as "+alias.asname if alias.asname else "") for alias in names))
    result='"""Shared mathematical helpers retained for the US research; unrelated study entry points excluded."""\n'+"\n\n".join(kept)+"\n"
    ast.parse(result)
    return result


def prune_auxiliary(files,roots):
    for _ in range(6):
        requested={path:set() for path in files if auxiliary(path)}
        for path,text in files.items():
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and not node.level:
                    target=module_path(node.module or "")
                    if target in requested:requested[target].update(alias.name for alias in node.names)
                elif isinstance(node,ast.Import):
                    for alias in node.names:
                        target=module_path(alias.name)
                        if target in requested:
                            local=alias.asname or alias.name
                            requested[target].update(item.attr for item in ast.walk(tree)
                                if isinstance(item,ast.Attribute) and isinstance(item.value,ast.Name) and item.value.id==local)
        updated=dict(files)
        for path,names in requested.items():
            if names:updated[path]=slim_shared_module(files[path],names)
        reachable=set();pending=list(roots)
        while pending:
            path=pending.pop()
            if path in reachable or path not in updated:continue
            reachable.add(path)
            pending.extend(dependency for name in imports(path,updated[path])
                           if (dependency:=module_path(name)) in updated)
        updated={path:text for path,text in updated.items() if path in reachable}
        if updated==files:return updated
        files=updated
    return files


def collect(book, shared=()):
    pending, found, external, changes = list(seeds(book)), {}, set(), []
    shared = set(shared)
    while pending:
        path = pending.pop()
        if path in found or path.as_posix() in shared:
            continue
        if book == "interday" and path.name == "calculate_five_currents_intraday_positions.py":
            continue
        if book == "intraday" and path.parts[0] == "five_currents_daily":
            continue
        text, changed = export_text(path, (ROOT/path).read_text(encoding="utf-8-sig"), book)
        found[path] = text; changes.extend(changed)
        for name in imports(path, text):
            dependency = module_path(name)
            if dependency is not None:
                if dependency.as_posix() not in shared:
                    pending.append(dependency)
            else:
                external.add(name.split(".")[0])
    return prune_auxiliary(found,seeds(book)), external, changes


def scan(files):
    findings = []
    token = json.loads((ROOT/"five_currents_dashboard/admin_access.json").read_text()).get("admin_token", "")
    for path, text in files.items():
        for number, line in enumerate(text.splitlines(), 1):
            if token and token in line:
                findings.append((str(path), number, "live dashboard token"))
            for match in URI.finditer(line):
                if urlsplit(match.group(0)).password:
                    findings.append((str(path), number, "password-bearing connection string"))
            if re.search(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{25,}|db-[A-Za-z0-9]{25,})", line):
                findings.append((str(path), number, "API credential pattern"))
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name) and SECRET_NAMES.search(target.id) and len(node.value.value)>12:
                        findings.append((str(path), node.lineno, "literal credential assignment"))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    interday = collect("interday")
    intraday = collect("intraday", shared={path.as_posix() for path in interday[0]})
    for book, (files, external, changes) in (("interday",interday),("intraday",intraday)):
        issues = scan(files)
        print(json.dumps({"book":book,"python_files":len(files),"third_party_imports":sorted(external),
                          "sanitized_literals":len(changes),"blocking_findings":issues},indent=2),flush=True)
        if issues:
            raise RuntimeError("Export blocked by credential scanner")
        if not args.write:
            continue
        destination = OUTPUT/f"five-currents-{book}"
        destination.mkdir(parents=True,exist_ok=True)
        previous=destination/"source_manifest.json"
        if previous.is_file():
            for item in json.loads(previous.read_text())["files"]:
                if Path(item["path"]) not in files:
                    stale=(destination/item["path"]).resolve()
                    if not stale.is_relative_to(destination.resolve()):raise ValueError("Unsafe export path")
                    if stale.is_file():
                        if digest(stale)!=item["export_sha256"]:raise ValueError("Previously exported file was edited: "+item["path"])
                        stale.unlink()
        inventory=[]
        for path, text in sorted(files.items()):
            target=destination/path;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(text,encoding="utf-8",newline="\n")
            inventory.append({"path":path.as_posix(),"source_sha256":digest(ROOT/path),"export_sha256":digest(target)})
        (destination/"source_manifest.json").write_text(json.dumps({"book":book,"files":inventory,
            "sanitization":changes,"private_data_included":False},indent=2),encoding="utf-8")
        if book=="interday":
            asset_destination=destination/"five_currents_daily/dashboard_assets"
            asset_destination.mkdir(parents=True,exist_ok=True)
            for name in ("index.html","app.css","app.js"):
                shutil.copyfile(ROOT/"five_currents_daily/dashboard_assets"/name,asset_destination/name)
            collector=ROOT/"maintenance/roll_safe_daily_20261028/ibkr_daily_curve_jobs.py"
            if collector.is_file():
                text,_=export_text(Path("ibkr_daily_curve_jobs.py"),collector.read_text(),book)
                (destination/"ibkr_daily_curve_jobs.py").write_text(text,encoding="utf-8")


if __name__=="__main__":
    main()
