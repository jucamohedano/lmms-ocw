"""Build per-task `<task_name>.json` reward metadata from ConceptNet + lm-eval tasks.

Steps (see docs/reward_design_v2.md): (1) enumerate labels from task objects,
(2) load filtered ConceptNet edges once, (3) extract parents / grandparents /
synonyms / attributes / closed-set siblings, (4) optional dataset-specific
fallbacks, (5) human curation of the emitted JSON.

Use ``--skip-att AtLocation`` (repeatable) from ``eval_model.py`` or this module's
CLI to omit an attribute relation from JSON and skip those edges during the
ConceptNet stream; the default pickle cache filename includes a ``.skip_*`` suffix
so caches for different skips do not collide.

This module mirrors ``verl.utils.reward_score.classification._normalise`` so
JSON keys match runtime lookups.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from src import utils

log = utils.get_logger(__name__, rank_zero_only=True)

# Relations kept after filtering (English, weight >= MIN_WEIGHT).
CN_RELATIONS = frozenset(
    {
        "/r/IsA",
        "/r/Synonym",
        "/r/HasProperty",
        "/r/HasA",
        "/r/AtLocation",
    }
)
MIN_WEIGHT = 1.0

# Reward JSON attribute keys (must match verl classification RELATIONS attribute names).
ALL_ATTR_KEYS: tuple[str, ...] = ("HasProperty", "HasA", "AtLocation")

# Map CLI / user input -> ConceptNet /r/... relation (for streaming skip).
REWARD_ATTR_TO_CN: dict[str, str] = {
    "HasProperty": "/r/HasProperty",
    "HasA": "/r/HasA",
    "AtLocation": "/r/AtLocation",
}

_CANONICAL_SKIP_ALIASES: dict[str, str] = {
    "hasproperty": "HasProperty",
    "hasa": "HasA",
    "atlocation": "AtLocation",
}

_URI_STEM_RE = re.compile(r"^(/c/[a-z]+/[^/]+)")


def _stem_uri(uri: str) -> str:
    """Return the /c/lang/term stem URI, stripping sense suffixes (e.g. ``/n``, ``/wn/...``).

    For example ``/c/en/abyssinian/n`` becomes ``/c/en/abyssinian``.
    """
    m = _URI_STEM_RE.match(uri)
    return m.group(1) if m else uri


# Outgoing adjacency: concept_uri -> relation -> list of (target_uri, weight)
Outgoing = dict[str, dict[str, list[tuple[str, float]]]]

# Optional per-task hooks: (label_norm) -> extra dict with optional keys
# parents, grandparents, synonyms (lists of str, reward-normalised)
DatasetFallback = Callable[[str], dict[str, object]]

DATASET_FALLBACKS: dict[str, DatasetFallback] = {}


def register_dataset_fallback(task_name: str, fn: DatasetFallback) -> None:
    """Register dataset-specific ConceptNet miss handling (e.g. StanfordCars)."""
    DATASET_FALLBACKS[task_name] = fn


def parse_skip_attributes(names: list[str] | None) -> tuple[frozenset[str], frozenset[str]]:
    """Validate ``--skip-att`` values and return (reward keys, ConceptNet /r/... relations)."""
    if not names:
        return frozenset(), frozenset()
    reward_skip: set[str] = set()
    for raw in names:
        key = raw.strip()
        if not key:
            continue
        canon = _CANONICAL_SKIP_ALIASES.get(key.lower())
        if canon is None:
            allowed = ", ".join(sorted(REWARD_ATTR_TO_CN))
            raise ValueError(f"Unknown --skip-att {raw!r}. Use one of: {allowed}.")
        reward_skip.add(canon)
    rf = frozenset(reward_skip)
    cnf = frozenset(REWARD_ATTR_TO_CN[k] for k in rf)
    return rf, cnf


def cache_suffix_for_skips(skip_reward_keys: frozenset[str]) -> str:
    """Filename fragment so pickle caches differ when different relations are omitted."""
    if not skip_reward_keys:
        return ""
    slug = "_".join(sorted(k.lower() for k in skip_reward_keys))
    return f".skip_{slug}"


def active_reward_attr_keys(skip_reward_attrs: frozenset[str]) -> frozenset[str]:
    """ConceptNet-backed attribute keys still used for metadata."""
    return frozenset(ALL_ATTR_KEYS) - skip_reward_attrs


# ---------------------------------------------------------------------------
# Normalisation — keep in sync with verl classification reward
# ---------------------------------------------------------------------------


def normalise_label(text: str) -> str:
    """Normalize label text the same way as the verl classification reward."""
    text = text.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def label_norm_to_concept_uri(label_norm: str) -> str:
    """Map reward key to primary English ConceptNet URI."""
    inner = label_norm.replace(" ", "_")
    return f"/c/en/{inner}"


def concept_uri_to_norm(uri: str) -> str:
    """Map /c/en/... (possibly with POS/sense suffix) to reward key."""
    stem = _stem_uri(uri)
    if not stem.startswith("/c/en/"):
        return ""
    rest = stem[6:]
    return normalise_label(rest.replace("_", " "))


# ---------------------------------------------------------------------------
# Task label enumeration (matches ttw_offline._extract_gt_label)
# ---------------------------------------------------------------------------


def _extract_gt_label(doc: object, task_obj: object) -> str:
    """Extract the ground-truth label string from a task document.

    Handles both string labels (caltech101, dtd, oxford_pets) and integer
    choice indices (some lm-eval tasks).
    """
    gt = task_obj.doc_to_target(doc)
    if isinstance(gt, int):
        choices = doc.get("choices", doc.get("options", []))
        gt = choices[gt] if choices else str(gt)
    return str(gt).strip()


def _get_docs_eval_split(task_obj: object) -> list:
    """Return documents from test, validation, or training splits."""
    if task_obj.has_test_docs():
        return list(task_obj.test_docs())
    if task_obj.has_validation_docs():
        return list(task_obj.validation_docs())
    if task_obj.has_training_docs():
        return list(task_obj.training_docs())
    if hasattr(task_obj, "dataset") and "test" in task_obj.dataset:
        return list(task_obj.dataset["test"])
    return []


def _get_train_docs(task_obj: object) -> list:
    """Return training documents, falling back to validation if unavailable."""
    if task_obj.has_training_docs():
        return list(task_obj.training_docs())
    if task_obj.has_validation_docs():
        return list(task_obj.validation_docs())
    if hasattr(task_obj, "dataset") and "train" in task_obj.dataset:
        return list(task_obj.dataset["train"])
    return []


def iter_task_docs_for_label_coverage(task_obj: object) -> Iterator[object]:
    """Yield documents from train and eval splits (labels may repeat)."""
    for doc in _get_train_docs(task_obj):
        yield doc
    for doc in _get_docs_eval_split(task_obj):
        yield doc


def enumerate_canonical_label_keys(task_obj: object) -> list[str]:
    """Sorted unique normalised keys used as JSON object keys for the reward."""
    keys: set[str] = set()
    for doc in iter_task_docs_for_label_coverage(task_obj):
        gt = _extract_gt_label(doc, task_obj)
        keys.add(normalise_label(gt))
    return sorted(keys)


# ---------------------------------------------------------------------------
# ConceptNet load (stream gzipped assertions, filter English + relations + weight)
# ---------------------------------------------------------------------------


def _parse_assertion_line(line: str) -> tuple[str, str, str, float] | None:
    """Return (relation, start_stem, end_stem, weight) or None if skipped."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        return None
    _uri, relation, start, end = parts[0], parts[1], parts[2], parts[3]
    json_blob = "\t".join(parts[4:])
    if relation not in CN_RELATIONS:
        return None
    if not start.startswith("/c/en/") or not end.startswith("/c/en/"):
        return None
    try:
        meta = json.loads(json_blob)
    except json.JSONDecodeError:
        return None
    weight = float(meta.get("weight", 1.0))
    if weight < MIN_WEIGHT:
        return None
    return relation, _stem_uri(start), _stem_uri(end), weight


def _quick_relation_column(line: str) -> str | None:
    """Return relation URI from first tab-separated fields without parsing JSON."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        return None
    return parts[1]


def stream_build_outgoing(
    gz_path: Path,
    log_every: int = 2_000_000,
    *,
    skip_cn_relations: frozenset[str] | None = None,
) -> Outgoing:
    """One pass over ``conceptnet-assertions-5.7.0.csv.gz`` → outgoing adjacency."""
    skip_cn_relations = skip_cn_relations or frozenset()
    outgoing: Outgoing = {}
    n_ok = 0
    n_lines = 0
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_lines += 1
            if log_every and n_lines % log_every == 0:
                log.info("ConceptNet: %s lines read, %s edges kept", f"{n_lines:,}", f"{n_ok:,}")
            if skip_cn_relations:
                rel = _quick_relation_column(line)
                if rel and rel in skip_cn_relations:
                    continue
            parsed = _parse_assertion_line(line)
            if parsed is None:
                continue
            relation, start, end, weight = parsed
            if start not in outgoing:
                outgoing[start] = {}
            if relation not in outgoing[start]:
                outgoing[start][relation] = []
            outgoing[start][relation].append((end, weight))
            n_ok += 1
    log.info("ConceptNet: done (%s lines, %s edges kept).", f"{n_lines:,}", f"{n_ok:,}")
    return outgoing


def load_or_build_outgoing_cache(
    gz_path: Path,
    cache_pkl: Path | None,
    *,
    rebuild: bool = False,
    skip_cn_relations: frozenset[str] | None = None,
) -> Outgoing:
    """Load pickled filtered graph, or stream from gz and optionally save."""
    skip_cn_relations = skip_cn_relations or frozenset()
    if cache_pkl and cache_pkl.is_file() and not rebuild:
        with open(cache_pkl, "rb") as f:
            return pickle.load(f)  # noqa: S301 — local ConceptNet cache only; not untrusted input.
    outgoing = stream_build_outgoing(gz_path, skip_cn_relations=skip_cn_relations)
    if cache_pkl:
        cache_pkl.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_pkl, "wb") as f:
            pickle.dump(outgoing, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Wrote ConceptNet cache -> %s", cache_pkl)
    return outgoing


# ---------------------------------------------------------------------------
# Metadata assembly for one label
# ---------------------------------------------------------------------------


def _unique_sorted(strings: Iterable[str]) -> list[str]:
    """Return a list of unique strings, sorted."""
    return sorted({s for s in strings if s})


def _outgoing_targets(outgoing: Outgoing, uri: str, relation: str) -> list[str]:
    """Return list of target URIs for a given relation from the outgoing adjacency."""
    return [t for t, _ in outgoing.get(uri, {}).get(relation, [])]


def _collect_attributes(
    uri: str,
    outgoing: Outgoing,
    active_keys: frozenset[str],
) -> dict[str, set[str]]:
    """Gather HasProperty / HasA / AtLocation tags for a single URI (only ``active_keys``)."""
    attrs: dict[str, set[str]] = {k: set() for k in ALL_ATTR_KEYS}
    rel_map = {"/r/HasProperty": "HasProperty", "/r/HasA": "HasA", "/r/AtLocation": "AtLocation"}
    for cn_rel, key in rel_map.items():
        if key not in active_keys:
            continue
        for target in _outgoing_targets(outgoing, uri, cn_rel):
            val = concept_uri_to_norm(target)
            if val:
                attrs[key].add(val)
    return attrs


def build_entry_for_label(
    label_norm: str,
    dataset_label_keys: set[str],
    outgoing: Outgoing,
    fallback: DatasetFallback | None,
    *,
    skip_reward_attrs: frozenset[str] | None = None,
) -> dict[str, object]:
    """Single label metadata dict matching classification reward expectations."""
    skip_reward_attrs = skip_reward_attrs or frozenset()
    active_keys = active_reward_attr_keys(skip_reward_attrs)
    uri = label_norm_to_concept_uri(label_norm)

    parents_raw = [concept_uri_to_norm(t) for t in _outgoing_targets(outgoing, uri, "/r/IsA")]
    parents_raw = [p for p in parents_raw if p and p != label_norm]

    synonyms_raw = [concept_uri_to_norm(t) for t in _outgoing_targets(outgoing, uri, "/r/Synonym")]
    synonyms_raw = [s for s in synonyms_raw if s and s != label_norm]

    grandparents: set[str] = set()
    for p_uri in _outgoing_targets(outgoing, uri, "/r/IsA"):
        for gp in _outgoing_targets(outgoing, p_uri, "/r/IsA"):
            gpn = concept_uri_to_norm(gp)
            if gpn and gpn != label_norm and gpn not in parents_raw:
                grandparents.add(gpn)

    combined = _collect_attributes(uri, outgoing, active_keys)
    for ancestor_norm in list(parents_raw) + list(grandparents):
        ancestor_uri = label_norm_to_concept_uri(ancestor_norm)
        ancestor_attrs = _collect_attributes(ancestor_uri, outgoing, active_keys)
        for key in ALL_ATTR_KEYS:
            combined[key] |= ancestor_attrs[key]

    attrs = {key: sorted(combined[key]) for key in ALL_ATTR_KEYS}

    parent_set = set(parents_raw)
    siblings: set[str] = set()
    for other in dataset_label_keys:
        if other == label_norm:
            continue
        other_uri = label_norm_to_concept_uri(other)
        other_parents = {
            concept_uri_to_norm(t) for t in _outgoing_targets(outgoing, other_uri, "/r/IsA")
        }
        other_parents.discard("")
        other_parents.discard(other)
        if parent_set & other_parents:
            siblings.add(other)

    entry: dict[str, object] = {
        "synonyms": _unique_sorted(synonyms_raw),
        "parents": _unique_sorted(parents_raw),
        "grandparents": _unique_sorted(grandparents),
        "siblings": _unique_sorted(siblings),
        "attributes": attrs,
    }

    if fallback is not None:
        extra = fallback(label_norm)
        for k, v in extra.items():
            if k == "parents" and isinstance(v, list):
                entry["parents"] = _unique_sorted(
                    list(entry["parents"]) + [normalise_label(x) for x in v]
                )
            elif k == "grandparents" and isinstance(v, list):
                entry["grandparents"] = _unique_sorted(
                    list(entry["grandparents"]) + [normalise_label(x) for x in v]
                )
            elif k == "synonyms" and isinstance(v, list):
                entry["synonyms"] = _unique_sorted(
                    list(entry["synonyms"]) + [normalise_label(x) for x in v]
                )
            elif k == "attributes" and isinstance(v, dict):
                for ak, av in v.items():
                    if ak in entry["attributes"] and isinstance(av, list):
                        merged = set(entry["attributes"][ak]) | {normalise_label(x) for x in av}
                        entry["attributes"][ak] = sorted(merged)

    return entry


def build_task_metadata_table(
    task_name: str,
    label_keys: list[str],
    outgoing: Outgoing,
    *,
    skip_reward_attrs: frozenset[str] | None = None,
) -> dict[str, object]:
    """Full JSON object: keys are normalised labels, values are metadata dicts."""
    skip_reward_attrs = skip_reward_attrs or frozenset()
    keys_set = set(label_keys)
    fallback = DATASET_FALLBACKS.get(task_name)
    table: dict[str, object] = {}
    for lk in label_keys:
        table[lk] = build_entry_for_label(
            lk, keys_set, outgoing, fallback, skip_reward_attrs=skip_reward_attrs
        )
    return table


def write_task_metadata_json(
    task_name: str,
    table: dict[str, object],
    out_dir: Path,
) -> Path:
    """Write the metadata table to a JSON file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


# ---------------------------------------------------------------------------
# Orchestration from eval / ttw_offline
# ---------------------------------------------------------------------------


def build_metadata_for_grpo_tasks(
    *,
    task_manager: object,
    task_names: list[str],
    conceptnet_gz: Path,
    cache_pkl: Path | None,
    metadata_out_dir: Path,
    rebuild_cache: bool = False,
    skip_attributes: list[str] | None = None,
    get_tasks_as_dict: Callable[..., dict] | None = None,
) -> list[Path]:
    """Load ConceptNet once, then emit one JSON per task. Returns written paths."""
    if get_tasks_as_dict is None:
        from src.data.tasks import get_tasks_as_dict as _gtd

        get_tasks_as_dict = _gtd

    reward_skip, cn_skip = parse_skip_attributes(skip_attributes)
    if reward_skip:
        log.info(
            "Skipping attribute relations in metadata (and ConceptNet stream): %s",
            sorted(reward_skip),
        )

    outgoing = load_or_build_outgoing_cache(
        conceptnet_gz,
        cache_pkl,
        rebuild=rebuild_cache,
        skip_cn_relations=cn_skip,
    )
    written: list[Path] = []
    for task_name in task_names:
        task_obj_dict = get_tasks_as_dict([task_name], task_manager)
        if not task_obj_dict or task_name not in task_obj_dict:
            log.warning("Skip metadata: failed to load task %s", task_name)
            continue
        task_obj = task_obj_dict[task_name]
        label_keys = enumerate_canonical_label_keys(task_obj)
        if not label_keys:
            log.warning("Skip metadata: no labels for %s", task_name)
            continue
        table = build_task_metadata_table(
            task_name, label_keys, outgoing, skip_reward_attrs=reward_skip
        )
        path = write_task_metadata_json(task_name, table, metadata_out_dir)
        written.append(path)
        log.info("Reward metadata: %d classes -> %s", len(label_keys), path)
    return written


def _parse_cli(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Build GRPO classification reward JSON from ConceptNet."
    )
    p.add_argument(
        "--tasks", required=True, help="Comma-separated lm-eval task names (e.g. oxford_pets)."
    )
    p.add_argument(
        "--conceptnet_assertions_gz",
        required=True,
        type=Path,
        help="Path to conceptnet-assertions-5.7.0.csv.gz",
    )
    p.add_argument(
        "--conceptnet_cache_pkl",
        type=Path,
        default=None,
        help="Pickle cache for filtered edges (large).",
    )
    p.add_argument(
        "--grpo_reward_metadata_dir",
        type=Path,
        default=Path("./reward_metadata"),
        help="Output directory for <task>.json",
    )
    p.add_argument(
        "--conceptnet_rebuild_cache", action="store_true", help="Ignore existing pickle cache."
    )
    p.add_argument(
        "--skip-att",
        dest="skip_attributes",
        action="append",
        default=None,
        metavar="REL",
        help=(
            "Repeatable. Omit this attribute from metadata JSON and skip its ConceptNet edges "
            "when streaming (HasProperty, HasA, or AtLocation). Example: --skip-att AtLocation."
        ),
    )
    p.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Optional lm-eval include_path for TaskManager.",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Model name hint for TaskManager (often unused).",
    )
    return p.parse_args(argv)


def main_cli(argv: list[str] | None = None) -> int:
    """Run the stand-alone ConceptNet metadata builder CLI."""
    args = _parse_cli(argv)
    from src.data.tasks import TaskManager, get_tasks_as_dict

    tm = TaskManager(include_path=args.include_path, model_name=args.model_name or "cli")
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    skip_list = args.skip_attributes or []
    reward_skip, _ = parse_skip_attributes(skip_list)
    cache_pkl = args.conceptnet_cache_pkl
    if cache_pkl is None:
        cache_pkl = args.grpo_reward_metadata_dir / (
            f".conceptnet_en_filtered{cache_suffix_for_skips(reward_skip)}.pkl"
        )
    build_metadata_for_grpo_tasks(
        task_manager=tm,
        task_names=task_names,
        conceptnet_gz=args.conceptnet_assertions_gz,
        cache_pkl=cache_pkl,
        metadata_out_dir=args.grpo_reward_metadata_dir,
        rebuild_cache=args.conceptnet_rebuild_cache,
        skip_attributes=skip_list,
        get_tasks_as_dict=get_tasks_as_dict,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
