"""
Dependency Length Analysis: Natural vs Randomized Word Orders
Across 12 typologically diverse languages from Universal Dependencies.

For each sentence:
 - Natural DL = sum of |i - head(i)| in the observed word order
 - Random DL = same, but with uniformly random word order
 - Projective Random DL = random order preserving projectivity

Hypothesis: Natural < Random Projective < Random
"""
import conllu
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

random.seed(42)
np.random.seed(42)

# ---- Configuration ----
DATA_DIR = Path("./ud_data")

LANGUAGES = [
    # (Language, treebank_dir, train_file, word_order, family)
    ("English", "UD_English-EWT", "en_ewt-ud-train.conllu", "SVO", "Indo-European"),
    ("Hindi", "UD_Hindi-HDTB", "hi_hdtb-ud-train.conllu", "SOV", "Indo-European"),
    ("Japanese", "UD_Japanese-GSD", "ja_gsd-ud-train.conllu", "SOV", "Japonic"),
    ("Arabic", "UD_Arabic-PADT", "ar_padt-ud-train.conllu", "VSO", "Afro-Asiatic"),
    ("Chinese", "UD_Chinese-GSD", "zh_gsd-ud-train.conllu", "SVO", "Sino-Tibetan"),
    ("Russian", "UD_Russian-SynTagRus", "ru_syntagrus-ud-train-a.conllu", "Free", "Indo-European"),
    ("Turkish", "UD_Turkish-IMST", "tr_imst-ud-train.conllu", "SOV", "Turkic"),
    ("Finnish", "UD_Finnish-TDT", "fi_tdt-ud-train.conllu", "Free", "Uralic"),
    ("Korean", "UD_Korean-GSD", "ko_gsd-ud-train.conllu", "SOV", "Koreanic"),
    ("Spanish", "UD_Spanish-GSD", "es_gsd-ud-train.conllu", "SVO", "Indo-European"),
    ("Basque", "UD_Basque-BDT", "eu_bdt-ud-train.conllu", "SOV", "Isolate"),
    ("Indonesian", "UD_Indonesian-GSD", "id_gsd-ud-train.conllu", "SVO", "Austronesian"),
]

MAX_SENTENCES = 2000  # cap per language for tractability
MIN_LEN, MAX_LEN = 5, 50  # filter sentence length
N_RANDOM = 10  # number of random shuffles to average per sentence
N_PROJECTIVE = 10  # number of projective shuffles to average per sentence


# ---- Sentence extraction ----
def extract_sentence(tokens):
    """
    Return a list of (id, head) pairs from a parsed CoNLL-U sentence.
    Skips multiword tokens (id like '3-4') and empty nodes (id like '3.1').
    Returns None if any structural problem (missing head, etc.).
    """
    pairs = []
    for tok in tokens:
        tid = tok["id"]
        # Only keep simple integer ids (real words)
        if not isinstance(tid, int):
            continue
        head = tok["head"]
        if head is None:
            return None
        pairs.append((tid, head))
    return pairs


def dependency_length(pairs, position_map=None):
    """
    Compute total dependency length.
    pairs: list of (id, head) where id and head are 1-indexed positions in original sentence.
    position_map: dict {original_id -> new_position}. If None, use original positions.
    Root dependencies (head==0) are excluded.
    """
    total = 0
    for tid, head in pairs:
        if head == 0:
            continue
        if position_map is None:
            total += abs(tid - head)
        else:
            total += abs(position_map[tid] - position_map[head])
    return total


# ---- Random shuffling ----
def random_position_map(n):
    """Uniform random permutation: original id i -> new position perm[i-1]+1"""
    perm = list(range(1, n + 1))
    random.shuffle(perm)
    return {i + 1: perm[i] for i in range(n)}


# ---- Projective random shuffling ----
def build_children(pairs):
    """Build dict of head -> list of children (including root=0)."""
    children = defaultdict(list)
    for tid, head in pairs:
        children[head].append(tid)
    return children


def projective_random_layout(pairs):
    """
    Generate a random projective linearization of the dependency tree.

    Algorithm: At each node, randomly permute (node + its children's subtrees)
    and recursively layout each subtree as a contiguous span.

    Returns: position_map {original_id -> new_position}
    """
    children = build_children(pairs)
    roots = children[0]  # nodes whose head is 0

    position_map = {}
    counter = [1]  # mutable counter for next position

    def layout(node):
        # Children of `node` plus `node` itself need to be ordered.
        # Each is a "slot": either the head word (size 1) or a child subtree.
        slots = ['HEAD'] + list(children.get(node, []))
        random.shuffle(slots)
        for s in slots:
            if s == 'HEAD':
                position_map[node] = counter[0]
                counter[0] += 1
            else:
                layout(s)

    # Layout each root tree (most sentences have exactly one root)
    random.shuffle(roots)
    for r in roots:
        layout(r)

    return position_map


# ---- Per-sentence analysis ----
def analyze_sentence(pairs):
    """Return dict with natural DL and averaged random/projective DLs."""
    n = len(pairs)
    natural = dependency_length(pairs)

    rand_dls = [dependency_length(pairs, random_position_map(n)) for _ in range(N_RANDOM)]
    proj_dls = [dependency_length(pairs, projective_random_layout(pairs)) for _ in range(N_PROJECTIVE)]

    return {
        "n": n,
        "natural": natural,
        "random": np.mean(rand_dls),
        "projective": np.mean(proj_dls),
    }


# ---- Per-language analysis ----
def analyze_language(name, path, word_order, family):
    print(f"  Processing {name} ({path.name})...")
    with open(path, encoding="utf-8") as f:
        sentences = list(conllu.parse_incr(f))

    # Optionally subsample BEFORE doing heavy work
    random.shuffle(sentences)

    results = []
    for sent in sentences:
        pairs = extract_sentence(sent)
        if pairs is None:
            continue
        n = len(pairs)
        if n < MIN_LEN or n > MAX_LEN:
            continue
        results.append(analyze_sentence(pairs))
        if len(results) >= MAX_SENTENCES:
            break

    df = pd.DataFrame(results)
    df["language"] = name
    df["word_order"] = word_order
    df["family"] = family
    print(f"    -> {len(df)} sentences analyzed")
    return df


# ---- Main ----
def main():
    print("Loading and analyzing treebanks...")
    all_dfs = []
    for name, repo, fname, wo, fam in LANGUAGES:
        path = DATA_DIR / repo / fname
        if not path.exists():
            print(f"  Skipping {name}: {path} not found")
            continue
        all_dfs.append(analyze_language(name, path, wo, fam))

    full = pd.concat(all_dfs, ignore_index=True)
    full.to_csv("sentence_results.csv", index=False)
    print(f"\nSaved {len(full)} sentence results to sentence_results.csv")

    # Per-language summary (means of per-sentence DL)
    summary = (full.groupby(["language", "word_order", "family"])
               .agg(n_sentences=("n", "count"),
                    avg_len=("n", "mean"),
                    natural=("natural", "mean"),
                    random=("random", "mean"),
                    projective=("projective", "mean"))
               .reset_index())
    summary["nat_vs_rand_pct"] = (1 - summary["natural"] / summary["random"]) * 100
    summary["nat_vs_proj_pct"] = (1 - summary["natural"] / summary["projective"]) * 100
    summary.to_csv("language_summary.csv", index=False)

    print("\n=== Per-language summary ===")
    print(summary.to_string(index=False))

    return full, summary


if __name__ == "__main__":
    main()
