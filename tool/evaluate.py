import sys
from datasets import load_metric
import evaluate

dir = str(sys.argv[1])

with open(dir+"/hyps.txt", "r") as r:
    predictions = r.readlines()
    
with open(dir+"/refs.txt", "r") as r:
    references = r.readlines()
    
meteor = evaluate.load('meteor')
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')

score = rouge.compute(predictions=predictions, references=references)
print("Rouge-1:", round(score["rougeL"].mid.fmeasure * 100, 2))
print("Rouge-L:", round(score["rougeL"].mid.fmeasure * 100, 2))

score = meteor.compute(predictions=predictions, references=references)
print("Meteor:", round(score * 100, 2))

score = bleu.compute(predictions=predictions, references=references, max_order=4, smooth=True)
print("BLEU-4:", round(score * 100, 2))
