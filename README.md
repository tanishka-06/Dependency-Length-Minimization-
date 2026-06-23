# Dependency Length Minimization Across 12 Languages

Do real sentences put related words closer together than random word orders would? I tested this across 12 languages — and the answer was yes, every time.

## TL;DR

- **Built:** A Python pipeline that parses dependency-grammar treebanks, computes word-distance ("dependency length") for natural sentences vs. randomized rewordings of the same sentence, and runs statistical tests on the result.
- **Tools:** Python, `conllu`, Pandas, NumPy, SciPy, Matplotlib.
- **Result:** Across 24,000 sentences in 12 languages (English, Hindi, Japanese, Arabic, Chinese, Russian, Turkish, Finnish, Korean, Spanish, Basque, Indonesian), real word order kept related words **51–72% closer together** than fully random word order — and still **9.5–40.8% closer** than random orderings that were forced to "look like" valid sentences. Every language, every test, statistically significant (p < 1e-100 in most cases).

Full write-up with all stats and figures: [`report/cgs410_Dependency_Length_Report.pdf`](report/cgs410_Dependency_Length_Report.pdf)

## Why this is interesting

Every sentence has a hidden structure: each word "points to" the word it depends on (e.g. an adjective points to the noun it describes). The farther apart two connected words sit, the harder a sentence is to hold in your head while parsing it. The question: do languages actually arrange words to minimize that distance, or is it just a side effect of grammar rules?

To test it, I took real sentences, shuffled their words into different orders (while keeping the same underlying grammar/dependencies), and compared how "spread out" the real sentence was vs. its shuffled versions. I used two kinds of shuffles — totally random, and a stricter version that's only allowed to produce orders that look grammatically plausible — to make sure the effect wasn't just "real sentences avoid obvious nonsense."

## How it works

1. **Data** — training sets from 12 [Universal Dependencies](https://universaldependencies.org/) treebanks (2,000 sentences/language, 24,000 total), covering 4 different word-order families (SVO, SOV, VSO, free) so the result isn't an English-specific quirk.
2. **Metric** — for each sentence, sum up the distance between every word and the word it depends on. Do this for (a) the real sentence, (b) 10 random reshuffles, and (c) 10 "structurally valid" reshuffles.
3. **Test** — paired Wilcoxon signed-rank test per language, comparing real vs. each shuffle type.

## Results

| Language | Real sentence | "Valid" random shuffle | Fully random shuffle | % shorter vs. valid shuffle | % shorter vs. random shuffle |
|---|---|---|---|---|---|
| Chinese | 85.3 | 94.2 | 212.0 | 9.5% | 59.8% |
| Turkish | 30.6 | 37.8 | 66.8 | 19.0% | 54.1% |
| Hindi | 70.0 | 87.0 | 171.7 | 19.5% | 59.2% |
| Finnish | 39.0 | 48.1 | 81.5 | 19.0% | 52.2% |
| Korean | 39.6 | 48.2 | 92.6 | 17.9% | 57.3% |
| Russian | 54.6 | 68.7 | 134.9 | 20.5% | 59.5% |
| English | 57.2 | 74.3 | 142.4 | 23.0% | 59.8% |
| Spanish | 74.9 | 104.4 | 230.9 | 28.2% | 67.5% |
| Indonesian | 54.7 | 76.7 | 159.3 | 28.6% | 65.7% |
| Arabic | 90.4 | 129.4 | 295.2 | 30.1% | 69.4% |
| Basque | 37.4 | 45.1 | 76.3 | 17.1% | 51.0% |
| Japanese | 57.6 | 97.3 | 206.1 | 40.8% | 72.0% |

(numbers = average dependency length per sentence — lower means words are packed tighter together)

The gap also gets bigger for longer sentences — real sentences scale roughly linearly with length, while fully-random shuffles scale almost quadratically (see `figures/03_dl_vs_length.png` after running the pipeline).

## Files

- `analyze.py` — parses the 12 treebanks, computes real/random/valid-random dependency lengths per sentence → `sentence_results.csv`, `language_summary.csv`
- `visualize.py` — runs the statistical tests and generates all charts → `figures/`, `statistical_tests.csv`
- `report/` — full written report (PDF) with the academic framing, theory, and discussion, if you want the deeper version
- `requirements.txt` — dependencies

## Running it

```bash
pip install -r requirements.txt
```

Download the 12 treebanks listed in `analyze.py` from [Universal Dependencies](https://universaldependencies.org/) into `./ud_data/<treebank_name>/`, then:

```bash
python analyze.py      # -> sentence_results.csv, language_summary.csv
python visualize.py    # -> figures/, statistical_tests.csv
```

## Limitations

- Capped at 2,000 sentences/language for runtime — more would tighten confidence intervals but wouldn't likely flip the direction of the result.
- Treebank annotation quality varies by language/source.
- Dependency length is one proxy for processing difficulty; it doesn't capture everything (e.g. surprisal, working-memory load from other sources).

## Background reading

This replicates and extends the dependency length minimization test from:
- Futrell, Mahowald & Gibson (2015), *Large-scale evidence of dependency length minimization in 37 languages*, PNAS.
- Gibson (1998), *Linguistic complexity: Locality of syntactic dependencies*, Cognition.
