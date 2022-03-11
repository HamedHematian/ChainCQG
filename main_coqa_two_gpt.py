import os
import time
import numpy as np
import pandas as pd
import tqdm
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, set_seed

import argparse

import os, sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

from utils import *

parser = argparse.ArgumentParser(description="ChainCQG")

parser.add_argument("--random-seed",
                    type=int,
                    default=2020)

parser.add_argument("--warmup-steps",
                    type=int)

parser.add_argument("--learning-rate",
                    type=float)

parser.add_argument("--batch-size",
                    type=int)

parser.add_argument("--gradient-accumulation-steps",
                    type=int)

parser.add_argument("--num-train-epochs",
                    type=int)

parser.add_argument("--do-train",
                    action="store_true")

parser.add_argument("--do-valid",
                    action="store_true")

parser.add_argument("--do-predict",
                    action="store_true")

parser.add_argument("--checkpoint-dir",
                    type=str,
                    default=None)

parser.add_argument("--max-target-length",
                    type=int,
                    default=64)

parser.add_argument("--top-p",
                    type=float,
                    default=0.2)

parser.add_argument("--top-k",
                    type=int,
                    default=400)

parser.add_argument("--temper",
                    type=float,
                    default=0.7)

parser.add_argument("--model-size",
                    type=str,
                    help="Model size")

parser.add_argument("--dataset-name",
                    type=str,
                    help="dataset to use")

parser.add_argument("--use-all-loss",
                    action='store_true',
                    help="Use base model or not")

parser.add_argument("--loss-discount",
                    type=float,
                    default=1.0,
                    help="the loss discount for turn before the final two turn, work when use_all_loss=True")

parser.add_argument("--fp16",
                    action="store_true")

args = parser.parse_args()

set_seed(args.random_seed)

tokenizer_dir = "data/coqa/tokenizer"
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)

# init all datasets
[train_dataset_dir, val_dataset_dir, test_dataset_dir] = get_dataset_dir_by_name(args.dataset_name)

train_data = torch.load(train_dataset_dir)
#train_data = train_data[:20]
val_data = torch.load(val_dataset_dir)
#val_data = val_data[:20]
test_data = torch.load(test_dataset_dir)


class TwoGPTDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
        self.__process__()
        
    def __len__(self):
        return len(self.data)
    
    def __process__(self):
        self.processed_data = []
        for one in self.data:
            one_dial_tokens = one
            one_role_ids = [idx%2 for idx in range(len(one_dial_tokens))]
            self.processed_data.append([one_role_ids, one_dial_tokens])
            
    def __getitem__(self, index):
        [role_ids, dial_tokens] = self.processed_data[index]
        #rold_ids = torch.Tensor(role_ids)
        #dial_tokens = torch.Tensor(dial_tokens)
        return role_ids, dial_tokens
    
    def collate(self, unpacked_data):
        return unpacked_data

train_dataset = TwoGPTDataset(train_data)
val_dataset = TwoGPTDataset(val_data)
test_dataset = TwoGPTDataset(test_data)

train_dataloader = DataLoader(dataset=train_dataset, 
                              shuffle=True, 
                              batch_size=args.batch_size,
                              collate_fn=train_dataset.collate)

val_dataloader = DataLoader(dataset=val_dataset, 
                            shuffle=False, 
                            batch_size=args.batch_size,
                            collate_fn=train_dataset.collate)

test_dataloader = DataLoader(dataset=test_dataset, 
                            shuffle=False, 
                            batch_size=args.batch_size,
                            collate_fn=train_dataset.collate)


# load the model
if args.model_size == "small":
    model_type = "gpt2" 
elif args.model_size == "medium":
    model_type = "gpt2-medium"
elif args.model_size == "large":
    model_type = "gpt2-large"
else:
    raise NotImplementedError()

model_A = GPT2LMHeadModel.from_pretrained(model_type)
model_B = GPT2LMHeadModel.from_pretrained(model_type)

model_A.resize_token_embeddings(len(tokenizer))
model_B.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda")
model_A = model_A.to(device)
model_B = model_B.to(device)
# model_B = model_A


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets, mask, label_smoothing=-1, reduce=None, loss_discount=1, end_length=-1):
        """
        reduce: None, "batch", "sentence"
        """
        return sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce, loss_discount, end_length)


def sequence_cross_entropy_with_logits(logits, targets, mask, label_smoothing, reduce, loss_discount=1.0, end_length=-1):
    # type: (Tensor, Tensor, Tensor, float, bool)-> Tensor
    """
    label_smoothing : ``float``, optional (default = 0.0)
        It should be smaller than 1.
    """

    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / float(num_classes)
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
                                       
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(-1, logits.shape[1])
    
    # shape : (batch, sequence_length)
    # batch size = 1
    # mask[0][:-20] = 0
    loss = negative_log_likelihood * mask
    # loss = loss.unsqueeze(0)
    # breakpoint()

    if reduce:
        # shape : (batch,)
        # breakpoint()
        #assert len(loss) == 1

        # only use in training time
        if end_length != -1 and reduce == "batch":
            
            # breakpoint()
            loss[:, :-end_length] = loss[:, :-end_length] * loss_discount
        
        loss = loss.sum(1) / (mask.sum(1) + 1e-13)
        
        if reduce == "batch":
            # shape : scalar
            loss = loss.mean()

    return loss


def train_one_iter(batch, update_count, fp16=False):
    role_ids, dialog_tokens = batch
    dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]
    
    past_key_values = None
    all_logits = []
    
    for turn_num, dial_turn_inputs in enumerate(dial_inputs):
        if role_ids[turn_num] == 0:
            # breakpoint()
            output = model_A(dial_turn_inputs, past_key_values=past_key_values)
            logits = output.logits
            past_key_values = output.past_key_values

            if args.use_all_loss:
                all_logits.append(logits)
            elif turn_num == len(dial_inputs) - 1 or turn_num == len(dial_inputs) - 2:
                all_logits.append(logits)
        else:
            # breakpoint()
            output = model_B(dial_turn_inputs, past_key_values=past_key_values)
            logits = output.logits
            past_key_values = output.past_key_values
            
            if args.use_all_loss:
                all_logits.append(logits)
            elif turn_num == len(dial_inputs) - 1 or turn_num == len(dial_inputs) - 2:
                all_logits.append(logits)

    # breakpoint()
    length = all_logits[-2].shape[1] + all_logits[-1].shape[1] - 1
    all_logits = torch.cat(all_logits, dim=1)
    
    # target
    all_logits = all_logits[:, :-1].contiguous()
    target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
    target_mask = torch.ones_like(target).float()
    # breakpoint()
    
    # only last two turn loss
    if args.use_all_loss:
        loss = criterion(all_logits, target[:, :], target_mask[:, :],
                         label_smoothing=0.02, reduce="batch",
                         loss_discount=args.loss_discount, end_length=length)
    else:
        # length = all_logits.shape[1]
        loss = criterion(all_logits, target[:, -length:], target_mask[:, -length:], 
                            label_smoothing=0.02, reduce="batch") 

    loss /= args.gradient_accumulation_steps

    loss.backward()
        
    record_loss = loss.item() * args.gradient_accumulation_steps
    perplexity = np.exp(record_loss)
    
    return record_loss, perplexity


def validate(dataloader):
    with torch.no_grad():
        pbar = progress_bar(dataloader)

        total_ppl = []

        for batch in pbar:
            
            if sum([len(item) for item in batch[0][1]]) > 1024:
                continue
            
            role_ids, dialog_tokens = batch[0]
            dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]

            past_key_values = None
            all_logits = []

            for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                if role_ids[turn_num] == 0:
                    output = model_A(dial_turn_inputs, past_key_values=past_key_values)
                    logits = output.logits
                    past_key_values = output.past_key_values
                    # all_logits.append(logits)
                    if turn_num == len(dial_inputs) - 2:
                        all_logits.append(logits)
                else:
                    output = model_B(dial_turn_inputs, past_key_values=past_key_values)
                    logits = output.logits
                    past_key_values = output.past_key_values
                    
                    if turn_num == len(dial_inputs) - 1:
                        all_logits.append(logits)
                        length_last_question = logits.shape[1]
            
            assert len(all_logits) == 2
            all_logits = torch.cat(all_logits, dim=1)
            
            # target
            all_logits = all_logits[:, :-1].contiguous()
            target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
            target_mask = torch.ones_like(target).float()
            
            loss = criterion(all_logits[:, -length_last_question:, :], target[:, -length_last_question:], \
                             target_mask[:, -length_last_question:], label_smoothing=-1, reduce="sentence")      
            # breakpoint()
            ppl = torch.exp(loss)
            total_ppl.extend(ppl.tolist())

        print(f"Epcoh {ep} Validation Perplexity: {np.mean(total_ppl)} Variance: {np.var(total_ppl)}")
        
        return np.mean(total_ppl)
    
def predict(dataloader):
    model_A.eval()
    model_B.eval()
    progress_bar = tqdm.tqdm
    pbar = progress_bar(dataloader)
    
    bos_token_id = tokenizer.encode(tokenizer.bos_token)[0]
    eos_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    
    results = []
    for batch in pbar:

        if sum([len(item) for item in batch[0][1]]) > 1024:
            continue
            
        role_ids, dialog_tokens = batch[0]
        dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]

        past_key_values = None

        for turn_num, dial_turn_inputs in enumerate(dial_inputs):
            if turn_num == len(dial_inputs)-1:
                break
            if role_ids[turn_num] == 0:
                output = model_A(dial_turn_inputs, past_key_values=past_key_values)
                past_key_values = output.past_key_values
            else:
                output = model_B(dial_turn_inputs, past_key_values=past_key_values)
                past_key_values = output.past_key_values
        
        output = model_B.generate(input_ids=torch.LongTensor([[bos_token_id]]).to(device),
                                  past_key_values=past_key_values,
                                  pad_token_id=eos_token_id,
                                  bos_token_id=bos_token_id,
                                  eos_token_id=eos_token_id,
                                  do_sample=True,
                                  max_length=args.max_target_length,
                                  top_p=args.top_p,
                                  top_k=args.top_k,
                                  temperature=args.temper)
        
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        label = tokenizer.decode(dial_inputs[-1][0].tolist(), skip_special_tokens=True)
        results.append({"result": pred, "label": label})  

    return results

if args.do_train:
    
    criterion = SequenceCrossEntropyLoss()

    # make Checkpoint dir
    if args.model_size == "small":
        _checkpoint_dir = args.dataset_name + "_small_Checkpoint"
    elif args.model_size == "medium":
        _checkpoint_dir = args.dataset_name + "_medium_Checkpoint"
    elif args.model_size == "large":
        _checkpoint_dir = args.dataset_name + "_large_Checkpoint"


    for i in range(1, 10):
        temp = _checkpoint_dir + "_" + str(i)
        if not os.path.isdir(temp):
            args.checkpoint_dir = temp
            break

    os.makedirs(args.checkpoint_dir, exist_ok=False)

    print(dict_to_text(args.__dict__))

    # store config for each checkpoint folder
    config_loc = args.checkpoint_dir + "/config.json"
    config = copy.deepcopy(args.__dict__)

    with open(config_loc, "w") as f:
        json.dump(config, f, indent=4)

    # define hyper-parameters
    num_train_optimization_steps = num_train_optimization_steps = len(train_dataset) * args.num_train_epochs // args.batch_size // args.gradient_accumulation_steps

    param_optimizer = list(model_A.named_parameters()) + list(model_B.named_parameters())
    no_decay = ['bias', 'ln', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]


    optimizer = AdamW(optimizer_grouped_parameters, 
                      lr=args.learning_rate,
                      eps=1e-06)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_optimization_steps)


    os.makedirs("models", exist_ok=True)


    update_count = 0
    progress_bar = tqdm.tqdm
    start = time.time()
    old_ppl = -float('Inf')

    for ep in range(args.num_train_epochs):

        "Training"
        pbar = progress_bar(train_dataloader)
        model_A.train()
        model_B.train()

        for batch in pbar:
            batch = batch[0]

            # without relative position, we skip dialogs
            #
            # if sum([len(item) for item in batch[1]]) > 1024:
            #   continue

            if sum([len(item) for item in batch[1]]) > 1024:
                continue

            record_loss, perplexity = train_one_iter(batch, update_count, fp16=args.fp16)

            update_count += 1

            if update_count % args.gradient_accumulation_steps == args.gradient_accumulation_steps - 1:
                # update for gradient accumulation
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # show progress
                pbar.set_postfix(loss=record_loss, perplexity=perplexity)

        "Evaluation"
        model_A.eval()
        model_B.eval()
        ppl = validate(val_dataloader)

        # save the model for later use
        filepath = os.path.join(args.checkpoint_dir, f"model_iter_{update_count}.pth")
        torch.save([model_A.state_dict(), model_B.state_dict()], filepath)
        print("saving at {}".format(filepath))
        
if args.do_predict:
    if not args.do_train:
        model_files = [f for f in os.listdir(args.checkpoint_dir) if f.startswith("model_iter_")]
        model_file = os.path.join(args.checkpoint_dir, sorted(model_files, reverse=True)[0])

        [model_A_state, model_B_state] = torch.load(model_file)
        model_A.load_state_dict(model_A_state)
        model_B.load_state_dict(model_B_state)
    
    results = predict(test_dataloader)
    prediction_file = os.path.join(args.checkpoint_dir, "predictions.json")
    print("saving result at {}.".format(prediction_file))
    with open(prediction_file, "w") as fout:
        json.dump(results, fout, indent=4)