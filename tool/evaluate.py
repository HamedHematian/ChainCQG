import sys
from datasets import load_metric

dir = str(sys.argv[1])

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def get_words(answer_text):
    answer_words = []
    prev_is_whitespace = True
    for c in answer_text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                answer_words.append(c)
            else:
                answer_words[-1] += c
            prev_is_whitespace = False
    return answer_words

predictions = []
references = []

with open(dir+"/hyps.txt", "r") as r:
    predictions = r.readlines()
    
with open(dir+"/refs.txt", "r") as r:
    references = r.readlines()

rouge = load_metric("rouge")
meteor = load_metric("meteor")
bleu = load_metric("bleu")

score = rouge._compute(predictions, references, rouge_types=["rougeL"])
print("RL:", round(score["rougeL"].mid.fmeasure*100, 2))

predictions = [get_words(sent) for sent in predictions]
references = [get_words(sent) for sent in references]

score = meteor._compute(predictions, references)
print("M:", round(score["meteor"]*100, 1))

references = [[sent] for sent in references]

score = bleu._compute(predictions, references, max_order=4, smooth=True)
print("BLEU:", round(score["bleu"]*100, 2))
