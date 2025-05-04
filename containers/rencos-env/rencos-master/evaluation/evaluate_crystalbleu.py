from collections import Counter
from nltk.util import ngrams
# 1. Import CrystalBLEU
from crystalbleu import corpus_bleu


def main(hyp, ref, len):
    with open(hyp, 'r') as r:
        hypothesis = r.readlines()
        res = {k: [" ".join(v.strip().lower().split()[:len])] for k, v in enumerate(hypothesis)}
    with open(ref, 'r') as r:
        references = r.readlines()
        gts = {k: [v.strip().lower()] for k, v in enumerate(references)}
        
        
    # 2. Extract trivially shared n-grams
    k = 500
    # <tokenized_corpus> is a list of strings
    # Extract all n-grams of length 1-4
    all_ngrams = []
    for n in range(1, 5):
        all_ngrams.extend(list(ngrams(res, n)))
    # Calculate frequencies of all n-grams
    frequencies = Counter(all_ngrams)
    trivially_shared_ngrams = dict(frequencies.most_common(k))

    # 3. Calculate CrystalBLEU
    weights = [
    (1,),
    (0.5, 0.5),
    (1/3, 1/3, 1/3),
    (0.25, 0.25, 0.25, 0.25)
    ]
    for i, weight in enumerate(weights):
        crystalBLEU_score = corpus_bleu(
            gts, res, weights=weight, ignoring=trivially_shared_ngrams)
        print(f"Bleu_{i+1}: {crystalBLEU_score}")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], eval(sys.argv[3]))