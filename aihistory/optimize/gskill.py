"""gskill for Python repositories: does a repository skill help an agent fix bugs here?

`optimize gskill init` copies your repository (without .git or virtualenvs), runs its tests, and
makes bug-fix tasks from procedural mutations of non-test modules: flipped comparisons, and/or
swaps, negated ifs, off-by-one constants, flipped boolean returns, +/- swaps, and removed
statements. A mutation becomes a task when it breaks between 1 and --max-f2p tests that passed
before. Single bugs go into disjoint test/val/train pools (no bug in two splits) and are combined
into --k-bug tasks, kept only when every bug still breaks one of its own tests.

Arms: `baseline` (no skill file), `control` (one line you write), and `gepa` (`optimize train`).
The skill is written as CLAUDE.md or AGENTS.md, whichever the agent reads. Grading restores the
pristine test files, then runs the suite: a run passes when every test that passed before
passes again.
"""

from __future__ import annotations

import ast
import fnmatch
import random
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .checks import clean_env

IGNORED = (".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv", "env",
           "node_modules", ".tox", "*.pyc", ".DS_Store")
DEFAULT_TESTS = ["test_*.py", "*_test.py", "conftest.py", "tests/*", "test/*"]
PYTEST_LINE = re.compile(r"^(PASSED|FAILED|ERROR) (\S+)")
UNITTEST_LINE = re.compile(r"^(\w+) \(([\w.]+)\)\s*\.\.\.\s*(ok|FAIL|ERROR|skipped.*|expected failure|unexpected success)\s*$")
UNITTEST_STATUS = {"ok": "PASSED", "FAIL": "FAILED", "ERROR": "ERROR", "unexpected success": "FAILED"}
SWAP_COMPARE = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
                ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
TASK_PROMPT = """{statement}

The repository is in the current directory. Run the tests with:

    {command}

The task is done when every test that passed before this change passes again.{baseline}"""
GENERATOR_SHAPES = (r"off[- ]by[- ]one", r"comparison operator", r"\bnegat\w* (?:an? )?(?:if|condition)",
                    r"\band\b\s*/\s*\bor\b|swap\w* (?:and|or)\b", r"\+\s*/\s*-|plus (?:and|or) minus",
                    r"removed statement|deleted (?:a )?(?:line|statement)", r"(?:boolean|bool) return")


@dataclass
class Mutation:
    file: str
    operator: str
    lineno: int
    end_lineno: int
    col: int
    end_col: int
    original: str
    replacement: str


def is_test_file(relative: str, tests: list[str]) -> bool:
    name = relative.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(relative, g) or fnmatch.fnmatch(name, g) or relative.startswith(g.rstrip("*"))
               for g in tests)


def copy_tree(source: Path, dest: Path, venv: Path | None = None) -> None:
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns(*IGNORED), dirs_exist_ok=True)
    if venv:
        (dest / ".venv").symlink_to(venv)


def run_suite(workspace: Path, command: str, timeout: int = 300) -> tuple[dict[str, str], str]:
    """({test id: PASSED | FAILED | ERROR}, output) from `pytest -rA` or `unittest -v` output."""
    env = {**clean_env(), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run(command, shell=True, cwd=workspace, capture_output=True, text=True, timeout=timeout, env=env)
        output = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return {}, f"the tests timed out after {timeout}s"
    outcomes: dict[str, str] = {}
    for line in output.splitlines():
        if match := PYTEST_LINE.match(line):
            status, test = match.groups()
        elif (match := UNITTEST_LINE.match(line)) and match.group(3) in UNITTEST_STATUS:
            name, where, result = match.groups()
            test, status = (where if where.endswith("." + name) else f"{where}.{name}"), UNITTEST_STATUS[result]
        else:
            continue
        if outcomes.get(test) not in ("FAILED", "ERROR"):  # a failing subtest wins over its parent
            outcomes[test] = status
    return outcomes, output


def passing(outcomes: dict[str, str]) -> set[str]:
    return {test for test, status in outcomes.items() if status == "PASSED"}


def failure_excerpt(output: str, workspace: Path, max_lines: int = 18) -> str:
    lines = output.splitlines()
    start = next((i for i, line in enumerate(lines) if re.match(r"^(?:FAIL|ERROR): |^_{3,} .+ _{3,}$", line)), None)
    if start is None:
        block = lines[-max_lines:]
    else:
        end = next((j for j in range(start + 1, len(lines)) if re.match(r"^(?:={5,}|-{5,}$|_{3,} )", lines[j])), len(lines))
        block = lines[start:end]
        if len(block) > max_lines:
            errors = [i for i, line in enumerate(block) if line.startswith("E ") or "Error" in line]
            first = max(1, (errors[0] if errors else 1) - 6)
            block = block[:1] + block[first:first + max_lines - 1]
    return "\n".join(block[:max_lines]).replace(str(workspace) + "/", "")


def _mutation(source: str, relative: str, operator: str, node: ast.AST, replacement: str) -> Mutation | None:
    original = ast.get_source_segment(source, node)
    if original is None or original == replacement:
        return None
    return Mutation(relative, operator, node.lineno, node.end_lineno, node.col_offset, node.end_col_offset, original, replacement)


def mutations(source: str, relative: str) -> list[Mutation]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in SWAP_COMPARE:
            swapped = ast.Compare(node.left, [SWAP_COMPARE[type(node.ops[0])]()], node.comparators)
            found.append(_mutation(source, relative, "compare_op", node, ast.unparse(swapped)))
        elif isinstance(node, ast.BoolOp):
            flipped = ast.BoolOp(ast.Or() if isinstance(node.op, ast.And) else ast.And(), node.values)
            found.append(_mutation(source, relative, "bool_op", node, ast.unparse(flipped)))
        elif isinstance(node, ast.If):
            test = ast.get_source_segment(source, node.test)
            if test and "\n" not in test:
                found.append(_mutation(source, relative, "negate_if", node.test, f"not ({test})"))
        elif isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 1000:
            found.append(_mutation(source, relative, "const_int", node, str(node.value + 1)))
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and type(node.value.value) is bool:
            found.append(_mutation(source, relative, "return_bool", node.value, str(not node.value.value)))
        elif isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub):
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                continue  # string concatenation: a minus would only raise TypeError
            swapped = ast.BinOp(node.left, ast.Sub() if isinstance(node.op, ast.Add) else ast.Add(), node.right)
            found.append(_mutation(source, relative, "binop", node, ast.unparse(swapped)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and len(node.body) > 2:
            for statement in node.body[1:]:
                if isinstance(statement, (ast.Assign, ast.AugAssign, ast.Expr)) and statement.lineno == statement.end_lineno:
                    if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)):
                        found.append(_mutation(source, relative, "remove_stmt", statement, "pass"))
    return [m for m in found if m is not None]


def apply_mutation(source: str, mutation: Mutation) -> str:
    lines = source.splitlines(keepends=True)
    head = lines[mutation.lineno - 1].encode()[:mutation.col].decode()  # ast offsets are UTF-8 bytes
    tail = lines[mutation.end_lineno - 1].encode()[mutation.end_col:].decode()
    return "".join(lines[:mutation.lineno - 1]) + head + mutation.replacement + tail + "".join(lines[mutation.end_lineno:])


def problem_statement(failing: int, excerpt: str) -> str:
    noun = "test fails" if failing == 1 else "tests fail"
    return (f"Bug report: after a recent change, {failing} {noun} that passed before. One of the failures:\n\n"
            f"```\n{excerpt}\n```\n\nFind and fix the bug in the source code. The tests are correct; do not modify test files.")


def task_prompt(statement: str, command: str, baseline_failures: list[str]) -> str:
    known = (f"\n\nThese tests already failed before the change and may stay failing: {', '.join(baseline_failures[:10])}"
             if baseline_failures else "")
    return TASK_PROMPT.format(statement=statement, command=command, baseline=known)


def _scratch(prefix: str, base: Path, venv: Path | None) -> Path:
    folder = Path(tempfile.mkdtemp(prefix=prefix))
    copy_tree(base, folder, venv)
    return folder


def single_bugs(base: Path, command: str, tests: list[str], venv: Path | None = None, candidates: int = 200,
                max_f2p: int = 20, workers: int = 4, seed: int = 7, sources: list[str] | None = None) -> tuple[set[str], set[str], list[dict]]:
    """(tests passing at base, tests failing at base, single-bug tasks)."""
    reference = _scratch("aihistory-gskill-", base, venv)
    try:
        outcomes, output = run_suite(reference, command)
    finally:
        shutil.rmtree(reference, ignore_errors=True)
    if not outcomes:
        raise ValueError(f"no test results from `{command}`; use `-m pytest -rA` or `-m unittest -v`. Output:\n{output[-600:]}")
    base_pass, base_fail = passing(outcomes), set(outcomes) - passing(outcomes)
    sites = []
    for path in sorted(base.rglob("*.py")):
        relative = path.relative_to(base).as_posix()
        if is_test_file(relative, tests) or (sources and not any(fnmatch.fnmatch(relative, g) for g in sources)):
            continue
        try:
            sites += mutations(path.read_text(encoding="utf-8"), relative)
        except (SyntaxError, UnicodeDecodeError):
            continue
    random.Random(seed).shuffle(sites)

    def evaluate(pair):
        index, mutation = pair
        mutated = apply_mutation((base / mutation.file).read_text(encoding="utf-8"), mutation)
        try:
            ast.parse(mutated)
        except SyntaxError:
            return None
        folder = _scratch(f"aihistory-gskill-{index}-", base, venv)
        try:
            (folder / mutation.file).write_text(mutated, encoding="utf-8")
            outcomes, output = run_suite(folder, command)
            broken = sorted(base_pass - passing(outcomes))
            if not outcomes or not 1 <= len(broken) <= max_f2p:
                return None
            return {**asdict(mutation), "mutated_source": mutated, "FAIL_TO_PASS": broken,
                    "statement": problem_statement(len(broken), failure_excerpt(output, folder))}
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = [r for r in pool.map(evaluate, enumerate(sites[:candidates])) if r]
    bugs, seen = [], set()
    for bug in results:
        key = (bug["file"], tuple(bug["FAIL_TO_PASS"]))
        if key not in seen:
            seen.add(key)
            bugs.append({"bug": f"b{len(bugs):03d}-{bug['operator']}", **bug})
    return base_pass, base_fail, bugs


def pool_sizes(bugs: int, k: int, fractions: dict[str, float]) -> dict[str, int]:
    """Bugs per split: by fraction, but at least `k` each when there are enough, so no split ends up empty."""
    names, sizes, left = list(fractions), {}, bugs
    for number, name in enumerate(names):
        if number == len(names) - 1:
            sizes[name] = left
            break
        reserve = k * (len(names) - number - 1)  # keep enough for the splits still to come
        sizes[name] = max(0, min(max(k, round(bugs * fractions[name])), left - reserve))
        left -= sizes[name]
    return sizes


def draw_combos(pool: list[dict], k: int, rng: random.Random, attempts: int = 4000) -> list[list[dict]]:
    """Distinct sets of `k` bugs, each from a different file: up to twice the pool size, since the
    suite later drops combinations in which a bug no longer shows."""
    combos, keys = [], set()
    for _ in range(attempts):
        if len(combos) >= 2 * len(pool) or len(pool) < k:
            break
        picked = rng.sample(pool, k)
        key = frozenset(b["bug"] for b in picked)
        if len({b["file"] for b in picked}) == k and key not in keys:
            keys.add(key)
            combos.append(picked)
    return combos


def combined_tasks(bugs: list[dict], base: Path, command: str, base_pass: set[str], base_fail: set[str], k: int = 3,
                   venv: Path | None = None, workers: int = 4, seed: int = 7,
                   fractions: dict[str, float] | None = None) -> dict[str, list[dict]]:
    """Disjoint pools of single bugs per split, combined `k` at a time from different files."""
    fractions = fractions or {"test": 0.4, "val": 0.2, "train": 0.4}
    rng = random.Random(seed)
    shuffled = list(bugs)
    rng.shuffle(shuffled)
    split, start = {}, 0
    for name, size in pool_sizes(len(shuffled), k, fractions).items():
        pool, start = shuffled[start:start + size], start + size
        combos = draw_combos(pool, k, rng)

        def build(pair, pool_name=name):
            index, picked = pair
            folder = _scratch(f"aihistory-gskill-{pool_name}-{index}-", base, venv)
            try:
                files = {b["file"]: b["mutated_source"] for b in picked}
                for relative, source in files.items():
                    (folder / relative).write_text(source, encoding="utf-8")
                outcomes, output = run_suite(folder, command)
                broken = sorted(base_pass - passing(outcomes))
                if not outcomes or not all(set(b["FAIL_TO_PASS"]) & set(broken) for b in picked):
                    return None
                statement = problem_statement(len(broken), failure_excerpt(output, folder))
                return {"id": f"{pool_name}-" + "+".join(b["bug"].split("-")[0] for b in picked),
                        "components": [b["bug"] for b in picked], "mutated_files": files, "FAIL_TO_PASS": broken,
                        "prompt": task_prompt(statement, command, sorted(base_fail)), "gold": [], "cls": f"{k}-bug"}
            finally:
                shutil.rmtree(folder, ignore_errors=True)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            built = [t for t in executor.map(build, enumerate(combos)) if t]
        split[name] = built[:max(1, len(pool))]
    return split


def generator_shapes(skill: str) -> list[str]:
    """Phrases in a learned skill that describe the task generator rather than the repository."""
    return [shape for shape in GENERATOR_SHAPES if re.search(shape, skill or "", re.IGNORECASE)]


def skill_document(skill: str) -> str | None:
    return "# Skills for this repository\n\n" + skill.strip() + "\n" if skill and skill.strip() else None


def restore_tests(workspace: Path, base: Path, tests: list[str]) -> None:
    """Put the pristine test files back, and remove test files the agent added."""
    for path in list(workspace.rglob("*")):
        relative = path.relative_to(workspace).as_posix()
        if path.is_file() and not path.is_symlink() and ".venv" not in path.parts and is_test_file(relative, tests) \
                and not (base / relative).exists():
            path.unlink()
    for path in base.rglob("*"):
        relative = path.relative_to(base).as_posix()
        if path.is_file() and is_test_file(relative, tests):
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def grade(workspace: Path, example: dict, base: Path, command: str, base_pass: set[str], tests: list[str]) -> dict:
    restore_tests(workspace, base, tests)
    outcomes, output = run_suite(workspace, command)
    now = passing(outcomes)
    broken = set(example["FAIL_TO_PASS"])
    still, regressions = sorted(broken - now), sorted((base_pass - broken) - now)
    changed = any((workspace / f).read_text(encoding="utf-8") != s for f, s in example["mutated_files"].items()
                  if (workspace / f).exists())
    if not outcomes:
        status = "suite_error"
    elif still:
        status = "f2p_failed" if changed else "bugged_file_untouched"
    elif regressions:
        status = "p2p_regression"
    else:
        status = "all_passed"
    return {"status": status, "still_failing": still[:5], "regressions": regressions[:5],
            "test_output": "" if status == "all_passed" else failure_excerpt(output, workspace)[:800]}


def init_run(out_dir: Path, name: str, repo: Path, command: str, venv: Path | None = None, tests: list[str] | None = None,
             sources: list[str] | None = None, k: int = 3, candidates: int = 200, max_f2p: int = 20, workers: int = 4,
             seed: int = 7, on_progress=None) -> Path:
    from ..util import private_dir, write_private_json
    from .experiment import load_config, run_path, save_config

    if not repo.is_dir():
        raise ValueError(f"no repository at {repo}")
    run_dir = private_dir(run_path(out_dir, name))
    config = load_config(run_dir) if (run_dir / "config.json").is_file() else {"arms": {}}
    if config.get("kind", "gskill") != "gskill":
        raise ValueError(f"{run_dir.name} is a {config['kind']} run; pick another --run name")
    base = run_dir / "base"
    shutil.rmtree(base, ignore_errors=True)
    copy_tree(repo, base)
    tests = tests or DEFAULT_TESTS
    base_pass, base_fail, bugs = single_bugs(base, command, tests, venv, candidates, max_f2p, workers, seed, sources)
    if on_progress:
        on_progress(f"{len(base_pass)} tests pass at the start; {len(bugs)} single-bug tasks from {candidates} candidates")
    if len(bugs) < 3 * k:
        raise ValueError(f"only {len(bugs)} usable bugs; try more --candidates, a lower --k, or check --tests")
    split = combined_tasks(bugs, base, command, base_pass, base_fail, k, venv, workers, seed) if k > 1 else \
        _singles_as_tasks(bugs, command, base_fail, seed)
    config.update({"kind": "gskill", "name": run_dir.name, "base": str(base), "test_command": command,
                   "venv": str(venv) if venv else None, "tests": tests, "base_pass": sorted(base_pass),
                   "base_fail": sorted(base_fail), "k": k})
    config["arms"].setdefault("baseline", {"skill": ""})
    save_config(run_dir, config)
    write_private_json(run_dir / "dataset.json", split)
    write_private_json(run_dir / "bugs.json", bugs)
    return run_dir


def _singles_as_tasks(bugs: list[dict], command: str, base_fail: set[str], seed: int) -> dict[str, list[dict]]:
    shuffled = list(bugs)
    random.Random(seed).shuffle(shuffled)
    tasks = [{"id": b["bug"], "components": [b["bug"]], "mutated_files": {b["file"]: b["mutated_source"]},
              "FAIL_TO_PASS": b["FAIL_TO_PASS"], "prompt": task_prompt(b["statement"], command, sorted(base_fail)),
              "gold": [], "cls": "1-bug"} for b in shuffled]
    n_test, n_val = round(len(tasks) * 0.4), round(len(tasks) * 0.2)
    return {"test": tasks[:n_test], "val": tasks[n_test:n_test + n_val], "train": tasks[n_test + n_val:]}


def allowed_tools(command: str) -> list[str]:
    return ["Read", "Edit", "Write", "Glob", "Grep", f"Bash({command}:*)", "Bash(ls:*)", "Bash(grep:*)",
            "Bash(sed -n:*)", "Bash(head:*)", "Bash(tail:*)", "Bash(wc:*)", "Bash(cat:*)"]

