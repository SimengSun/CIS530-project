# # MONOLINGUAL
python3 baselines.py --pairfile data/data/en_test/STS.input.track5.en-en.txt --predfile data/data/en_test/STS.pred.track5.en-en.txt --v 1

python3 evaluate.py --goldfile data/data/en_test/STS.gs.track5.en-en.txt --predfile data/data/en_test/STS.pred.track5.en-en.txt

# # MULTILINGUAL
# python3 baselines.py --pairfile data/data/es-en-test/STS.input.track4a.es-en.txt --predfile data/data/es-en-test/STS.pred.track4a.es-en.txt --v 2

# python3 evaluate.py --goldfile data/data/es-en-test/STS.gs.track4a.es-en.txt --predfile data/data/es-en-test/STS.pred.track4a.es-en.txt
cp baselines.py evaluate.py data.md scoring.md simple-baseline.md sub.sh writeup.pdf submission/
cd submission
zip ../submission.zip *