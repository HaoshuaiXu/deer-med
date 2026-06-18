# GENIA stratified low-resource train samples

- Source: `data/genia_term_corpus/convert/train.json`
- Seed: `42`
- Sampling unit: document
- Target distribution: entity mention type distribution from the full train set
- Guarantee: every entity type present in the full train set appears at least once
- Sample sizes: `100, 200, 500, 1000`

Each `{N}.json` file has a sibling `{N}_metadata.json` with type counts,
target proportions, sampled proportions, and L1 distribution distance.
