import json
import os
import argparse
import numpy as np
import random
import tqdm

from utils import *

from transformers import GPT2Tokenizer

parser = argparse.ArgumentParser(description="ChainCQG_prepro")

parser.add_argument("--dataset-name",
                  type=str,
                  default="coqa_two_gpt",
                  help="input type")

parser.add_argument("--model-name",
                  type=str,
                  help="pretrained model name for tokenizer")

parser.add_argument("--max-seq-length",
                  type=int,
                  default=512,
                  help="Model max input sequence length")

parser.add_argument("--min-context-length",
                  type=int,
                  default=32,
                  help="Model max input sequence length")

args = parser.parse_args()

tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
special_tokens_dict = {'additional_special_tokens':['<HL>', '</HL>', '<SEP>']}
tokenizer.add_special_tokens(special_tokens_dict)

tokenizer_dir = "data/coqa/tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)

print(f"Save the tokenizer into {tokenizer_dir}...")
tokenizer.save_pretrained(tokenizer_dir)

[train_dataset_dir, val_dataset_dir, test_dataset_dir] = get_dataset_dir_by_name(args.dataset_name)

train_data_file = "data/coqa/coqa-train-wikipedia.json"
val_data_file = "data/coqa/coqa-dev-wikipedia.json"

def get_features(data_file, tokenizer):
    hl_start_token_id = tokenizer.encode("<HL>")
    hl_end_token_id = tokenizer.encode("</HL>")
    sep_token_id = tokenizer.encode("<SEP>")
    bos_token_id = tokenizer.encode(tokenizer.bos_token)
    eos_token_id = tokenizer.encode(tokenizer.eos_token)
    
    with open(data_file, "r") as fin:
        data_list = json.load(fin)["data"]
    
    fid = 0
    features = []
    for data in tqdm.tqdm(data_list, total=len(data_list)):
        passage = data["story"]

        questions = sorted(data["questions"], key=lambda x: x["turn_id"])
        answers = sorted(data["answers"], key=lambda x: x["turn_id"])

        qa_history = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            feature = []
            rationale_start = answer["span_start"]
            rationale_end = answer["span_end"]

            forward_context = passage[:rationale_start]
            rationale_span = passage[rationale_start:rationale_end+1]
            backward_context = passage[rationale_end+1:]

            forward_tokens = tokenizer.encode(forward_context)
            rationale_tokens = tokenizer.encode(rationale_span)
            backward_tokens = tokenizer.encode(backward_context)

            answer_text = answer["input_text"]
            question_text = question["input_text"]

            answer_tokens = tokenizer.encode(answer_text)
            question_tokens = tokenizer.encode(question_text)

            qa_history.append((answer_tokens.copy(), question_tokens.copy()))

            assert len(tokenizer._tokenize(question_text)) <= args.max_seq_length

            first_answer = qa_history[0][0]
            ## <HL> </HL> <SEP> <EOS>
            max_passage_length = args.max_seq_length - (len(rationale_tokens)+len(first_answer)+4)
            
            if max_passage_length < args.min_context_length:
                print("answer is too long")
                continue
            
            if len(forward_tokens)+len(backward_tokens) > max_passage_length:
                while len(forward_tokens)+len(backward_tokens) > max_passage_length:
                    forward_tokens = forward_tokens[1:]
                    backward_tokens = backward_tokens[:-1]
            
            PH = forward_tokens + hl_start_token_id + rationale_tokens + hl_end_token_id + backward_tokens
            
            first_input = PH + sep_token_id + first_answer + eos_token_id
            assert len(first_input) <= args.max_seq_length, f"len:{len(first_input)}"
            feature.append(first_input)
            
            for turn_id, (a, q) in enumerate(qa_history):
                if turn_id != 0:
                    answer_tokens = a + eos_token_id
                    feature.append(answer_tokens)
                question_tokens = bos_token_id + q + eos_token_id
                feature.append(question_tokens)
                
            if fid < 3:
                print(tokenizer.decode(feature[-1]))
            fid += 1
            features.append(feature)

    return features

train_features = get_features(train_data_file, tokenizer)
valid_features = get_features(val_data_file, tokenizer)
  
torch.save(train_features, train_dataset_dir)
torch.save(valid_features, val_dataset_dir)
torch.save(valid_features, test_dataset_dir)

print("Done")