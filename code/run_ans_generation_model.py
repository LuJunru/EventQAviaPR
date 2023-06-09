# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, University of Warwick, Department of Computer Science. (authors: Junru Lu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, T5TokenizerFast, BartTokenizerFast, T5ForConditionalGeneration, BartForConditionalGeneration
from transformers import PegasusForConditionalGeneration
from utils import *
from optimization import *
from pathlib import Path
import re
from collections import Counter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                                  Path.home() / '.pytorch_pretrained_roberta'))

def unlikelihood_loss(decoder_input_ids, logits, weight_mask, selective_penalty=False):
    """
    reference: https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts/blob/main/modeling/finetune.py#L152
    decoder_input_ids - (N, s)
    logits      - (N, s, vocab_size)
    weight_mask - (N, vocab_size)
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    neg_probs = 1.0 - probs

    # replace zeros with small positive constant for stability
    neg_probs += (neg_probs == 0).float() * 1e-8
    log_neg_probs = torch.log(neg_probs)  # (N, s, v)

    # now create attention mask and apply it
    attention_mask = decoder_input_ids.eq(1).eq(0).float()
    attention_mask = attention_mask.unsqueeze(2).expand(-1, -1, logits.shape[2])
    log_neg_probs_masked = log_neg_probs * attention_mask

    # apply weight vector to the log probability tensor
    N, s = logits.size()[:2]
    weight_mask_expanded = weight_mask.unsqueeze(1)
    weighted_probs = log_neg_probs_masked * weight_mask_expanded

    if selective_penalty:
        indices = torch.argmax(logits, dim=-1)
        indices_mask = torch.nn.functional.one_hot(indices, num_classes=logits.shape[-1])  # (N, s, v)
        weighted_probs *= indices_mask

    return -torch.sum(weighted_probs)
    
def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="pre-trained model selected in the list: "
                             "allenai/unifiedqa-t5-{small, base, large ...} ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        required=True,
                        help="unique identifier for data file")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=320, # 8 * 8 * 5
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--sub_sample",
                        default=0, # 8 * 8 * 5
                        type=int,
                        help="0 means full data; otherwise, use K random sample for training")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--finetune",
                        action='store_true',
                        help="Whether to finetune LM.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--pos_weight',
                        type=int,
                        default=1,
                        help="positive weight on label 1")
    parser.add_argument("--load_model",
                        type=str,
                        help="cosmos_model.bin, te_model.bin",
                        default="")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mlp_hid_size",
                        default=64,
                        type=int,
                        help="hid dimension for MLP layer.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--cuda',
                        type=str,
                        default="",
                        help="cuda index")
    parser.add_argument('--save_model',
                        action='store_true',
                        help="save best or not")
    parser.add_argument('--device_num',
                        type=str,
                        default="0",
                        help="cuda device number")
    parser.add_argument('--penalty_weight_nonanswer',
                        type=float,
                        default=1.5,
                        help="penalty weight of non answer event")
    parser.add_argument('--penalty_weight_answer',
                        type=float,
                        default=0.0,
                        help="penalty weight of answer event")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    # args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    # fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.load_model:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    logger.info("current task is " + str(task_name))

    # construct model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if "t5" in args.model:
        model = T5ForConditionalGeneration.from_pretrained(args.model)
        tokenizerfast = T5TokenizerFast.from_pretrained(args.model)
    if "bart" in args.model:
        model = BartForConditionalGeneration.from_pretrained(args.model)
        tokenizerfast = BartTokenizerFast.from_pretrained(args.model)

    model.to(device)
    if args.do_train:
        train_data = load_data(args.data_dir, "final_train_ans_gen", args.file_suffix)
        if args.sub_sample > 0:
            random.Random(args.seed).shuffle(train_data)
            train_data = train_data[:args.sub_sample]
        train_features = convert_to_features(train_data, tokenizer, 
                                             max_length=args.max_seq_length, output_dir=args.output_dir, tokenizerfast=tokenizerfast, sep_tok=";")

        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'mask_ids'), dtype=torch.long)

        all_labels = select_field(train_features, 'labels')
        encoded_outputs = tokenizer(all_labels, padding=True, truncation=True, return_tensors="pt")
        all_output_ids = encoded_outputs['input_ids']
        all_output_mask = encoded_outputs['attention_mask']
        all_key_indices = torch.tensor(list(range(len(all_labels))), dtype=torch.long)

        event_ids_ = torch.tensor(select_field(train_features, 'event_ids'), dtype=torch.long)
        clr_ids_ = torch.tensor(select_field(train_features, 'clr_ids'), dtype=torch.long)
        sent_ids_ = torch.tensor(select_field(train_features, 'sent_ids'), dtype=torch.long)
        all_q_start = torch.tensor(select_field(train_features, 'q_start'))
        all_q_end = torch.tensor(select_field(train_features, 'q_end'))

        logger.info("id_size: {}, mask_size: {}, instance_key_size: {}, label_size: {}, \
                     event_ids_size: {}, sent_size: {}".format(
                     all_input_ids.size(), all_input_mask.size(), all_key_indices.size(), all_output_ids.size(), 
                     event_ids_.size(), sent_ids_.size()))

        train_data = TensorDataset(all_input_ids, all_input_mask, all_key_indices, all_output_ids, all_output_mask, 
                                   event_ids_, clr_ids_, sent_ids_, all_q_start, all_q_end)

        # free memory
        del train_features

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        model.train()

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        output_perf_file = os.path.join(args.output_dir, "dev_perf.txt")
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        t_total = num_train_steps

        if args.fp16:
            try:
                from apex.optimizers import FusedAdam
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False)
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=t_total)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        best_eval_f1 = 0.0
        best_eval_em = 0.0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss, tr_acc_start, tr_acc_end = 0.0, 0.0, 0.0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_masks, instance_indices, output_ids, output_masks, event_ids, clr_ids, sent_ids, q_start, q_end = batch
                sub_loss = None
                if 't5' in args.model:
                    if "PR" in args.output_dir:
                        outputs = model(input_ids, attention_mask=input_masks, labels=output_ids, decoder_attention_mask=output_masks)
                        loss = outputs.loss
                        logits = outputs.logits

                        # UL loss
                        weight_vector = torch.zeros((logits.size(0), logits.size(-1))).to(logits.device)
                        e_weight_0, e_weight_1 = args.penalty_weight_nonanswer, args.penalty_weight_answer
                        for b_i in range(logits.size(0)):  # For every sample
                            cur_c_0, cur_c_1 = [], []
                            b_sent_i = [bsi for bsi in sent_ids[b_i].tolist() if bsi[0] != -1]  # (S, 2)
                            for fs_i, fs_b_sent in enumerate(b_sent_i):  # For every sentence
                                for oe_ii, oe_i in enumerate(event_ids[b_i][fs_i]):  # For every possible event in one sentence
                                    if oe_i == -1:  # padding
                                        break
                                    cl_i = clr_ids[b_i][fs_i][oe_ii]  # event type
                                    if cl_i == 1:
                                        cur_c_1.append(input_ids[b_i, oe_i])  # answer event (or include neighbor tokens?)
                                    if cl_i == 0:
                                        cur_c_0.append(input_ids[b_i, oe_i])  # non answer event
                            weight_vector[b_i] = 1.0 / (logits.size(-1) + (e_weight_1 - 1) * len(cur_c_1) + \
                                                        (e_weight_0 - 1) * len(cur_c_0))  # equal penalty weights for all other tokens
                            if len(cur_c_0) != 0:
                                for cur_cc_0 in cur_c_0:
                                    weight_vector[b_i, cur_cc_0] *= e_weight_0  # higher penalty weights for non answer events
                            if len(cur_c_1) != 0:
                                for cur_cc_1 in cur_c_1:
                                    weight_vector[b_i, cur_cc_1] *= e_weight_1  # lower penalty weights on answer events, which means encourage of these tokens
                        decoder_input_ids = model.module.decoder._shift_right(output_ids)
                        ul_loss = unlikelihood_loss(decoder_input_ids, logits, weight_vector, True)
                        ul_loss /= decoder_input_ids.size(0)
                        ul_loss_weighted = ul_loss * 0.1
                        loss += ul_loss_weighted
                    else:
                        loss, _, _, final_hidden_state = model(input_ids, attention_mask=input_masks, labels=output_ids,  
                                                               decoder_attention_mask=output_masks)
                else:
                    output = model(input_ids, attention_mask=input_masks, labels=output_ids, 
                                   decoder_attention_mask=output_masks)
                    loss = output[0]

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += output_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if step > 0 and step % 100 == 0:
                    logger.info("current train loss is %s" % (tr_loss / float(nb_tr_steps)))

            if args.do_eval:
                eval_data = load_data(args.data_dir, "final_dev", "_ans_gen.json")
                eval_features = convert_to_features(eval_data, tokenizer,
                                                    max_length=args.max_seq_length, evaluation=True, 
                                                    output_dir=args.output_dir, tokenizerfast=tokenizerfast, sep_tok=";")

                eval_inputs = select_field(eval_features, 'inputs')
                eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                eval_input_mask = torch.tensor(select_field(eval_features, 'mask_ids'), dtype=torch.long)

                eval_event_ids_ = torch.tensor(select_field(eval_features, 'event_ids'), dtype=torch.long)
                eval_clr_ids_ = torch.tensor(select_field(eval_features, 'clr_ids'), dtype=torch.long)
                eval_sent_ids_ = torch.tensor(select_field(eval_features, 'sent_ids'), dtype=torch.long)

                eval_labels = select_field(eval_features, 'labels')
                eval_encoded_outputs = tokenizer(eval_labels, padding=True, truncation=True, return_tensors="pt")
                eval_output_ids = eval_encoded_outputs['input_ids']
                eval_key_indices = torch.tensor(list(range(len(eval_labels))), dtype=torch.long)

                eval_q_start = torch.tensor(select_field(eval_features, 'q_start'))
                eval_q_end = torch.tensor(select_field(eval_features, 'q_end'))

                eval_events = select_field(eval_features, 'events')
                eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_key_indices, eval_output_ids,
                                          eval_event_ids_, eval_clr_ids_, eval_sent_ids_, eval_q_start, eval_q_end)

                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                preds, golds, events = [], [], []
                model.eval()

                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_masks, instance_indices, output_ids, event_ids, clr_ids, sent_ids, q_start, q_end = batch

                    with torch.no_grad():
                        res = model.module.generate(input_ids, attention_mask=input_masks)
                        preds.extend([x.split(";") for x in tokenizer.batch_decode(res, skip_special_tokens=True)])
                        golds.extend([eval_labels[x].split(";") for x in instance_indices.tolist()])
                        events.extend([eval_events[x] for x in instance_indices.tolist()])

                ems, F1, recl, prec, F1_e, r_e, pr_e = [], [], [], [], [], [], []
                for pred, gold, event in zip(preds, golds, events):

                    # construct unigram counter
                    pred_counter = Counter([x for s in pred for x in re.sub(r'[^\w\s]', '', s).split(' ')])
                    gold_counter = Counter([x for s in gold for x in re.sub(r'[^\w\s]', '', s).split(' ')])

                    rl, pr, f1 = compute_unigram_f1(pred_counter, gold_counter)
                    F1.append(f1)
                    recl.append(rl)
                    prec.append(pr)

                    rl, pr, f1 = compute_event_f1(pred, event)
                    F1_e.append(f1)
                    r_e.append(rl)
                    pr_e.append(pr)

                    if f1 == 1.0:
                        ems.append(1.0)
                    else:
                        ems.append(0.0)

                logger.info("Event EM is %.4f" % np.mean(ems))
                logger.info("Token Recall, Precision, F1 are %.4f, %.4f, %.4f" %
                      (np.mean(recl), np.mean(prec), np.mean(F1)))
                logger.info("Event Recall, Precision, F1 are %.4f, %.4f, %.4f" %
                      (np.mean(r_e), np.mean(pr_e), np.mean(F1_e)))

                if np.mean(F1_e) > best_eval_f1:
                    best_eval_f1 = np.mean(F1_e)
                    logger.info("Save at Epoch %s" % epoch)
                    with open(output_perf_file, 'w') as outfile:
                        outfile.write("%.4f" % best_eval_f1)
                    if args.save_model:
                        torch.save(model_to_save.state_dict(), output_model_file)
                # if np.mean(ems) > best_eval_em:
                #     best_eval_em = np.mean(ems)
                #     logger.info("Save at Epoch %s" % epoch)
                #     with open(output_perf_file, 'w') as outfile:
                #         outfile.write("%.4f" % best_eval_em)
                #     if args.save_model:
                #         torch.save(model_to_save.state_dict(), output_model_file)

                model.train()

if __name__ == "__main__":
    main()
