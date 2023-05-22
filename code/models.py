from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import torch
import random
from torch import nn
from transformers import RobertaConfig, RobertaModel, BertPreTrainedModel

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

class RobertaSpanPredictor(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config, mlp_hid=16):
        super(RobertaSpanPredictor, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, 2)
        self.act = nn.Tanh()
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids)
        final_hidden_state = outputs[0]
        outputs = final_hidden_state.reshape(-1, final_hidden_state.size()[-1])
        out = self.act(self.linear1(self.dropout(outputs)))
        logits = self.linear2(out).reshape(input_ids.size()[0], input_ids.size()[1], -1)
        return logits, final_hidden_state

class RobertaSpanPredictor_PR(RobertaSpanPredictor):
    def __init__(self, config, mlp_hid=16):
        super(RobertaSpanPredictor_PR, self).__init__(config, mlp_hid)
        self.miu_score = torch.nn.Linear(config.hidden_size, 1)  # q event & non-q event -> score; 1 for Q&A, -1 for Q&~A
        self.h_attention_p_0 = torch.nn.MultiheadAttention(config.hidden_size, 1, batch_first=True)  # ~q sent as query, q sent as key and value
        self.h_score_p_proj_0 = torch.nn.Linear(config.hidden_size, 1)  # (batch, p len, 1024) -> (batch, p len, 1)
        self.h_attention_p_1 = torch.nn.MultiheadAttention(config.hidden_size, 1, batch_first=True)  # ~q sent as query, q sent as key and value
        self.h_score_p_proj_1 = torch.nn.Linear(config.hidden_size, 1)  # (batch, p len, 1024) -> (batch, p len, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                      qloc_s=None, qloc_e=None, sent_i=None, event_ids=None, clr_ids=None, eval_type=None):
        """
        eval_type: "event" or "all", different eval mode. "event" stands for prediction over only event tokens, while "all" stands for over all tokens.
        """
        final_hidden_state = self.roberta(input_ids, attention_mask=attention_mask, 
                                          token_type_ids=token_type_ids, position_ids=position_ids)[0]
        hp_score = torch.zeros((final_hidden_state.size(0), final_hidden_state.size(1), 2)).float().to(final_hidden_state.device)  # (batch, p len, 2)
        atten_scores_0 = []  # (batch * sent num, 1)
        atten_scores_1 = []  # (batch * sent num, 1)
        atten_scores_label = []  # (batch * sent num, 1); (1 or -1)
        miu_scores = []
        miu_scores_label = []

        # sent_ids: [[0, 10], [10, 17], [17, 40], [40, 109], [109, 180], [-1, -1]] -> (B, Sm, 2)
        # event_ids: [[5, -1, -1], [11, 14, -1], [-1, -1, -1], [42, 89, 91], [-1, -1, -1]] -> (B, Sm, Es)
        # clr_ids: [[2, -1, -1], [1, 0, -1], [-1, -1, -1], [0, 0, 0], [-1, -1,- 1]] -> (B, Sm, Es)
        # used in training, ignored in inference. In inference, predict over all tokens / all event tokens for event prediction.
        # miu_score: predict whether an event token is answer event or non-answer event. Corresponding labels are 1, -1.
        # u_score: regularized score to adjust posterior distribution

        """
        for events in one sent:
            miu_score: B X 1024 -> B X 1
            uu_score_0: miu_score * (B, 1) -> B X 1  # for 0 dim prediction
            uu_score_1: miu_score * (B, 1) -> B X 1  # for 1 dim prediction
        """

        for b_i in range(final_hidden_state.size(0)):  # For every sample
            b_sent_i = [bsi for bsi in sent_i[b_i].tolist() if bsi[0] != -1]  # (S, 2)
            final_q = final_hidden_state[b_i:b_i + 1, qloc_s[b_i]:qloc_e[b_i], :]  # (1, q len, 1024)
            
            hs_score_p_0 = self.h_attention_p_0(final_hidden_state[b_i:b_i + 1, :, :], final_q, final_q)[0]  # (1, p len, 1024)
            hs_score_p_0 = self.act(self.h_score_p_proj_0(hs_score_p_0))  # (1, p len, 1)
            hs_score_p_1 = self.h_attention_p_1(final_hidden_state[b_i:b_i + 1, :, :], final_q, final_q)[0]  # (1, p len, 1024)
            hs_score_p_1 = self.act(self.h_score_p_proj_1(hs_score_p_1))  # (1, p len, 1)

            if eval_type == "all":
                for fs_i, fs_b_sent in enumerate(b_sent_i):  # For every sentence
                    uu_score = [torch.tensor([[0.0, 0.0]]).to(hp_score.device)]  # (1, 2)
                    for t_i in range(fs_b_sent[0], fs_b_sent[1]):  # For every token in one sentence
                        hs_score_b = hs_score_p[:, t_i, :]
                        miu_score = self.miu_score(final_hidden_state[b_i:b_i + 1, t_i, :])
                        miu_scores.append(miu_score)
                        uu_score.append(miu_score * hs_score_b)
                    uu_score = sum(uu_score) / len(uu_score)
                    hp_score[b_i:b_i + 1, fs_b_sent[0]:fs_b_sent[1], :] = uu_score
            else:
                for fs_i, fs_b_sent in enumerate(b_sent_i):  # For every sentence
                    uu_score_0 = [torch.tensor([[0.0]]).to(hp_score.device)]  # (1, 1)
                    uu_score_1 = [torch.tensor([[0.0]]).to(hp_score.device)]  # (1, 1)
                    for oe_ii, oe_i in enumerate(event_ids[b_i][fs_i]):  # For every possible event in one sentence
                        if oe_i == -1:  # padding
                            break
                        cl_i = clr_ids[b_i][fs_i][oe_ii]  # type of event token
                        if cl_i == 2 or cl_i == -1:  # question event token or padding
                            break
                        hs_score_b_0 = hs_score_p_0[:, oe_i, :]
                        hs_score_b_1 = hs_score_p_1[:, oe_i, :]
                        miu_score = self.miu_score(final_hidden_state[b_i:b_i + 1, oe_i, :])
                        miu_scores.append(miu_score)
                        uu_score_0.append(miu_score * hs_score_b_0)  # for 0 dim
                        uu_score_1.append(miu_score * hs_score_b_1)  # for 1 dim
                        if cl_i == 1:
                            miu_scores_label += [1]
                        else:
                            miu_scores_label += [-1]
                    uu_score_0 = sum(uu_score_0) / len(uu_score_0)
                    uu_score_1 = sum(uu_score_1) / len(uu_score_1)
                    # atten_scores += uu_score.gather(-1, torch.abs(uu_score).max(dim=-1)[1].unsqueeze(-1)).reshape(-1).tolist()
                    atten_scores_0.append(uu_score_0)
                    atten_scores_1.append(uu_score_1)
                    if 1 in clr_ids[b_i][fs_i]:  # if current sentence contain any answer event tokens
                        atten_scores_label += [1]
                    else:
                        atten_scores_label += [-1]
                    hp_score[b_i:b_i + 1, fs_b_sent[0]:fs_b_sent[1], 0:1] = uu_score_0  # (1, s len, 1), broadcast event value to entrie sentence (or only event tokens?)
                    hp_score[b_i:b_i + 1, fs_b_sent[0]:fs_b_sent[1], 1:] = uu_score_1  # (1, s len, 1), broadcast event value to entrie sentence (or only event tokens?)
        outputs = final_hidden_state.reshape(-1, final_hidden_state.size()[-1])
        out = self.act(self.linear1(self.dropout(outputs)))
        logits = self.linear2(out).reshape(input_ids.size()[0], input_ids.size()[1], -1)
        logits = logits * torch.exp(hp_score)  # (bastch, p len, 2)

        sub_loss = None

        if eval_type is None:
            sub_loss = []
            subloss_fct = torch.nn.MSELoss()
            if len(atten_scores_0) > 0:
                atten_score_loss = subloss_fct(torch.tensor(atten_scores_0).to(logits.device), 
                                               torch.tensor(atten_scores_label).to(logits.device)) + \
                                   subloss_fct(torch.tensor(atten_scores_1).to(logits.device), 
                                               torch.tensor([-al for al in atten_scores_label]).to(logits.device))
                sub_loss.append(0.1 * atten_score_loss)
            if len(miu_scores) > 0:
                miu_score_loss = subloss_fct(torch.tensor(miu_scores).to(logits.device),
                                                torch.tensor(miu_scores_label).to(logits.device))
                sub_loss.append(0.1 * miu_score_loss)

            if len(sub_loss) > 0:
                sub_loss = sum(sub_loss)
            else:
                sub_loss = torch.tensor(0.0).to(logits.device)

        return logits, final_hidden_state, sub_loss
