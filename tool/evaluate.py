import sys
from datasets import load_metric
import evaluate
import os

def compute_scores(filename, epoch):
    
    dir = '../drive/MyDrive/CQG/Log/'
    out_file = '../drive/MyDrive/CQG/Log/scores.txt'
    filename = os.path.join(dir, filename)
    with open(filename, "r") as reader:
        items = json.load(reader)
        for item in items:
            results.append(item["result"])
            labels.append(item["label"])

    with open(dir+"/hyps.txt", "r") as r:
        predictions = r.readlines()

    with open(dir+"/refs.txt", "r") as r:
        references = r.readlines()

    meteor = evaluate.load('meteor')
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    # compute scores
    score = rouge.compute(predictions=predictions, references=references)
    score = meteor.compute(predictions=predictions, references=references)
    score = bleu.compute(predictions=predictions, references=references, max_order=4, smooth=True)
    
    out_txt = ''
    out_txt += f'Rouge-1:{round(score["rouge1"].mid.fmeasure * 100, 2)}\n'
    out_txt += f'Rouge-L:{round(score["rougeL"].mid.fmeasure * 100, 2)}\n'
    out_txt += f'Meteor:{round(score["meteor"] * 100, 2)}\n'
    out_txt += f'BLEU-4:{round(score["bleu"] * 100, 2)}\n'
    
    with open(outfile, 'a') as f:
        f.write('-------- EPOCH {} Evaluation --------\n')
        f.write(out_txt)
        f.write('\n')
    print(out_txt)
