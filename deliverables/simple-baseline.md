## Monolingual and multilingual semantic textual similarity

An example for monlingual semantic textual similarity is
`python3 simple-baseline.py --pairfile ../data/en-train.txt --predfile ../data/pred_en.txt --v 1`

`python3 evaluate.py --goldfile ../data/en-train.txt --predfile ../data/pred_en.txt`

The final baseline correlation (for this monolingual example) is 0.6265.

In `sub.sh`, I also provide a version for es-en where google translate takes 26 seconds to run on our machines:
`python3 simple-baseline.py --pairfile ../data/es-en.txt --predfile ../data/pred_es-en.txt --v 2`
with baseline Pearson correlation 0.5637.
(Examples are also written in `sub.sh`)