import sys
import json

dir=str(sys.argv[1])

results = []
labels = []

with open(dir+"/predictions.json", "r") as reader:
    items = json.load(reader)
    for item in items:
        results.append(item["result"])
        labels.append(item["label"])

with open(dir+"/hyps.txt", "w") as w:
    for line in results:
        w.write(line+"\n")

with open(dir+"/refs.txt", "w") as w:
    for line in labels:
        w.write(line+"\n")
