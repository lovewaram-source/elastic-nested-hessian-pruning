"""Export a small, self-contained source tree for GitHub, using only stdlib.

Historical experiments and local results remain in the working directory.
The public tree contains current entry points, their local dependencies,
selected tests, documentation and the original license. Never overwrites.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    "README.md", "INSTALL.md", "LICENSE", "CONTRIBUTING.md", ".gitignore",
    "requirements.txt", "docs/CODE_MAP.md", "docs/EXPERIMENTS.md", "docs/RESULTS.md",
    "docs/METHOD.md", "docs/RELEASING.md",
    "docs/GETTING_STARTED.md", "docs/VALIDATION.md",
    "docs/SORTING.md",
    "docs/figures/method-overview.svg", "docs/figures/ppl-comparison.svg",
)
RUNNERS = (
    "run_checkpoint_token_isolation", "run_rate_aligned_nested",
    "run_token_residual_diagnostics", "run_compensation_conflict",
    "benchmark_mean_hessian", "benchmark_module_token_optimization",
    "summarize_checkpoint_token_isolation", "summarize_replacement_error_experiment",
    "summarize_compensation_conflict", "export_github",
    "plot_readme_figures",
    "run_sorting_nested",
)
TESTS = (
    "data", "data_paths", "elastic_2of4", "nm_pruning", "experiment_common",
    "checkpoint_token_isolation",
    "mean_hessian", "module_token_isolation", "module_token_optimization",
    "replacement_error_routing", "token_residual_diagnostics", "compensation_conflict",
    "export_github",
    "readme_figures",
    "run_metadata", "sorting_runner",
)
# High-confidence credential patterns. Errors report paths, never matched values.
SECRET_RULES = (
    r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{30,}\b",
    r"\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b",
    r"\bAKIA[A-Z0-9]{16}\b",
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
    r'''(?i)\b(?:password|passwd)\s*=\s*["'][^"'\n]{4,}["']''',
)


def checked_source(root, relative):
    path = root / relative
    path.resolve().relative_to(root.resolve())
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked source: {relative}")
    return path


def local_dependency(root, current, module, level=0):
    """Resolve workspace imports without importing torch or executing modules."""
    parts = module.split(".") if module else []
    if level:
        base = current.parent
        for _ in range(level - 1):
            base = base.parent
        bases = [base]
    elif parts[:1] == ["rate_aligned_src"] and len(parts) > 1:
        bases, parts = [root / "lib"], parts[1:]
    else:
        bases = [current.parent, root, root / "scripts", root / "tests"]
    for base in bases:
        stem = base.joinpath(*parts)
        for path in (stem.with_suffix(".py"), stem / "__init__.py"):
            if path.is_file():
                return path.relative_to(root).as_posix()
    return None


def select_files(root, seeds=None):
    root = Path(root).resolve()
    if seeds is None:
        seeds = [*DOCUMENTS, "scripts/rate_aligned_src/__init__.py"]
        seeds += [f"scripts/{name}.py" for name in RUNNERS]
        seeds += [f"tests/test_{name}.py" for name in TESTS]
    selected, pending = set(), list(seeds)
    while pending:
        relative = pending.pop()
        if relative in selected:
            continue
        source = checked_source(root, relative)
        selected.add(relative)
        if source.suffix != ".py":
            continue
        tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [(alias.name, 0) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "", node.level)]
                names += [(".".join(filter(None, (node.module, alias.name))), node.level)
                          for alias in node.names if alias.name != "*"]
            else:
                continue
            for module, level in names:
                dependency = local_dependency(root, source, module, level)
                if dependency and dependency not in selected:
                    pending.append(dependency)
    return sorted(selected)


def export_project(root, destination, files, make_zip=False):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    archive = destination.with_suffix(".zip")
    if destination == root or destination in root.parents:
        raise ValueError("destination must not replace the source workspace")
    if destination.exists() or (make_zip and archive.exists()):
        raise FileExistsError("destination or archive exists; choose a fresh destination")
    # Check every source before creating any public files.
    payloads = {}
    for relative in sorted(set(files)):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe relative source path: {relative}")
        payload = checked_source(root, relative).read_bytes()
        content = payload.decode("utf-8-sig")
        if any(re.search(rule, content) for rule in SECRET_RULES):
            raise ValueError(f"possible credential in {relative}; export aborted")
        payloads[relative] = payload
    destination.mkdir(parents=True)
    hashes = {}
    for relative, payload in payloads.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        hashes[relative] = hashlib.sha256(payload).hexdigest()
    manifest = dict(format_version=1, source_file_count=len(hashes), sha256=hashes,
                    scope="Hessian experiments and their required kernels, tests and docs",
                    excludes="models, datasets, raw results, caches, local archives and git history")
    (destination / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if make_zip:
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
            for path in sorted(destination.rglob("*")):
                if path.is_file():
                    stream.write(path, path.relative_to(destination).as_posix())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "dist/github")
    parser.add_argument("--zip", action="store_true", help="also create a sibling .zip")
    args = parser.parse_args()
    manifest = export_project(ROOT, args.destination, select_files(ROOT), args.zip)
    size = sum((args.destination / p).stat().st_size for p in manifest["sha256"])
    print(f"Exported {manifest['source_file_count']} source files ({size / 1024:.1f} KiB)")
    print(args.destination.resolve())
    if args.zip:
        print(args.destination.resolve().with_suffix(".zip"))


if __name__ == "__main__":
    main()
