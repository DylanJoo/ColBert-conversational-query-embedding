import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from typing import Optional, Union, Dict, Any
from .loss import InBatchKLLoss, InBatchNegativeCELoss, PairwiseCELoss
from .colbert import ColBert

class TctColBertForCQE(ColBert):
    """ColBert for Conversational Query Embeddings 
    This class provides:
    (1) Train like ColBert (pairwise CE Loss)
    (2) Train like ColBert efficiently (In-batch negative Loss)
    (3) TCT training (In-bathc negative Loss + KL Loss)
    """
    def __init__(self, config, **kwargs):
        super(TctColBertForCQE, self).__init__(config, **kwargs)

        # pooler_type ,loss_type ,skiplist ,similarity_metric ,dim ,temperature ,linear
        self.freeze_document_encoder = kwargs.pop('freeze_document_embedding', True)
        self.kd_teacher = kwargs.pop('kd_teacher', None)

    def forward(self,
                q_input_ids: Optional[torch.Tensor] = None,
                q_attention_mask: Optional[torch.Tensor] = None,
                q_token_type_ids: Optional[torch.Tensor] = None,
                d_input_ids: Optional[torch.Tensor] = None,
                d_attention_mask: Optional[torch.Tensor] = None,
                d_token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                keep_d_dims: bool = True,
                **kwargs):
        """
        In this TctColBert model, using shared bert model as D & Q's encoder
        
        Note that if we use kd, 
        we will have a fixed teacher model with specified parameter settings
        """

        # Query pooling
        q = self.bert(
            input_ids=q_input_ids,
            attention_mask=q_attention_mask,
            token_type_ids=q_token_type_ids,
            **kwargs
        )
        if self.pooler_type == 'colbert': # futurewarning: deprecated
            Q = self.colbert_pooler(q.last_hidden_state)
        elif self.pooler_type == 'mean':
            Q = self.avg_pooler(q.last_hidden_state, q_attention_mask)

        # Document pooling 
        if self.freeze_document_encoder is False:
            d = self.bert(
                input_ids=d_input_ids,
                attention_mask=d_attention_mask,
                token_type_ids=d_token_type_ids,
                **kwargs
            )
            if self.pooler_type == 'colbert':
                d_mask = self.mask(d_input_ids)
                D = self.colbert_pooler(d.last_hidden_state, d_mask, keep_d_dims)
            elif self.pooler_type == 'mean':
                D = self.avg_pooler(d.last_hidden_state, d_attention_mask)
        else:
            output_teacher = self.kd_teacher.forward(
                    q_input_ids,
                    q_attention_mask, 
                    q_token_type_ids,
                    d_input_ids,
                    d_attention_mask,
                    d_token_type_ids,
                    **kwargs
            )
            scores_teacher = output_teacher['score']
            D = output_teacher['D']

        # KD
        if self.loss_type == "inbatch-KD" and self.freeze_document_encoder is False:
            scores_teacher = self.kd_teacher.forward(
                    q_input_ids,
                    q_attention_mask, 
                    q_token_type_ids,
                    d_input_ids,
                    d_attention_mask,
                    d_token_type_ids,
                    **kwargs
            )['score']

        
        # loss computing
        if self.loss_type == 'pairwise': 
            scores = self.pairwise_score(Q, D)
            loss = PairwiseCELoss(scores)
        elif self.loss_type == 'inbatch-negative' and self.pooler_type == 'colbert':
            scores = self.inbatch_score(Q, D)
            loss = InBatchNegativeCELoss(scores)
        elif self.loss_type == 'inbatch-negative' and self.pooler_type == 'mean':
            scores = Q @ D.permute(1, 0) # (B 2*B)
            loss = InBatchNegativeCELoss(scores)
        elif self.loss_type == 'inbatch-KD':
            scores = Q @ D.permute(1, 0) # (B 2*B)
            loss_ce = InBatchNegativeCELoss(scores)
            loss_kl = InBatchKLLoss(scores, scores_teacher, self.temperature)
            loss = self.gamma*loss_ce + (1-self.gamma)*loss_kl
            print(f"CE loss: {loss_ce*self.gamma} = {loss_ce} * {self.gamma}")
            print(f"KL loss: {loss_kl*(1-self.gamma)} = {loss_kl} * {(1-self.gamma)}")
        else:
            raise ValueError('Invalid loss type')

        return {'score': scores, 'loss': loss, 'Q': Q, 'D': D}
