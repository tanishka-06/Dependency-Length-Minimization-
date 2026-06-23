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

- Natural word order produced shorter dependency lengths than both random baselines in all 12 languages.
- Compared to unrestricted random word order, dependency lengths were reduced by 51–72%.
- Even against structurally valid random reorderings, dependency lengths were 9.5–40.8% shorter.
- The effect was statistically significant for every language tested (paired Wilcoxon signed-rank test; p < 1e-100 for most languages).
- The advantage of natural word order increased as sentence length grew.

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
