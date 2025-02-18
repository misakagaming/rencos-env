from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from meteor.meteor import Meteor
import numpy as np
import sys
import statistics

def main(directory, len, total):
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    
    rouge = []
    meteor = []
    cider = []
    for i in range(1, total+1):
        hyp = directory + "predictions.out-" + str(i) + ".txt"
        ref = directory + "trgs.given-" + str(i) + ".txt"
        with open(hyp, 'r') as r:
            hypothesis = r.readlines()
            res = {k: [" ".join(v.strip().lower().split()[:len])] for k, v in enumerate(hypothesis)}
        with open(ref, 'r') as r:
            references = r.readlines()
            gts = {k: [v.strip().lower()] for k, v in enumerate(references)}

        score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)
        bleu1.append(np.mean(scores_Bleu[0]))
        bleu2.append(np.mean(scores_Bleu[1]))
        bleu3.append(np.mean(scores_Bleu[2]))
        bleu4.append(np.mean(scores_Bleu[3]))

        score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
        meteor.append(score_Meteor)
        print("Meteor: "), score_Meteor

        score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
        rouge.append(score_Rouge)
        print("ROUGe: "), score_Rouge

        score_Cider, scores_Cider = Cider().compute_score(gts, res)
        cider.append(score_Cider)
        print("Cider: "), score_Cider
    print("BLEU-1: \n"), bleu1
    print("BLEU-2: \n"), bleu2
    print("BLEU-3: \n"), bleu3
    print("BLEU-4: \n"), bleu4
    print("ROUGE: \n"), rouge
    print("METEOR: \n"), meteor
    print("Cider: \n"), cider

if __name__ == '__main__':
    main(sys.argv[1],eval(sys.argv[2]), eval(sys.argv[3]))