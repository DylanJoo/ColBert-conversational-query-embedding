import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from typing import Optional, Union, Dict, Any
from .loss import InBatchKLLoss, InBatchNegativeCELoss, PairwiseCELoss

class ColBert(BertPreTrainedModel):
    def __init__(self, config, **kwargs):

        super(ColBert, self).__init__(config)

        self.pooler_type = kwargs.pop('pooler_type', 'colbert')
        self.loss_type = kwargs.pop('loss_type', 'inbatch_negative')
        
        # ColBert parameters
        self.skiplist = {}
        if kwargs.pop('mask_punctuation', True):
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {
                    w: True for symbol in string.punctuation for w in \
                            [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]
            }
        self.similarity_metric = kwargs.pop('similarity_metric', 'cosine')
        self.dim = kwargs.pop('dim', 128)
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)

        # TctColBert parameters
        self.gamma = kwargs.pop('gamma', 0.1)
        self.temperature = kwargs.pop('temperature', 0.25)

        self.bert = BertModel(config)
        self.init_weights()

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
        In this ColBert model, using shared bert model as D's & Q's encoder
        
        Note colbert adopt pairwise ranking loss,
        hence q_input_ids would be repetitve at first dimension, 
        while d_input_ids would concatenate by pos and neg document
        """

        # contextualized embeddgins
        q = self.bert(
            input_ids=q_input_ids,
            attention_mask=q_attention_mask,
            token_type_ids=q_token_type_ids,
            **kwargs
        )
        d = self.bert(
            input_ids=d_input_ids,
            attention_mask=d_attention_mask,
            token_type_ids=d_token_type_ids,
            **kwargs
        )

        # pooling
        if self.pooler_type == 'colbert':
            d_mask = self.mask(d_input_ids)
            Q = self.colbert_pooler(q.last_hidden_state)
            D = self.colbert_pooler(d.last_hidden_state, mask=d_mask, keep_dims=keep_d_dims)
        elif self.pooler_type == 'mean':
            Q = self.avg_pooler(q.last_hidden_state, q_attention_mask)
            D = self.avg_pooler(d.last_hidden_state, d_attention_mask)
        else:
            raise ValueError('Invalid pooler_type.')

        # loss computing 
        if self.loss_type == 'pairwise':
            scores_pairwise = self.pairwise_score(Q, D)  # (2*B 1)
            loss = PairwiseCELoss(scores_pairwise)
            return {'score': scores_pairwise, 'loss': loss, "Q": Q, "D": D}
        elif self.loss_type == 'inbatch-negative' and self.pooler_type == 'colbert':
            scores_inbatch = self.inbatch_score(Q, D) # (B 2*B) by Maxsim
            loss = InBatchNegativeCELoss(scores_inbatch)
            return {'score': scores_inbatch, 'loss': loss, "Q": Q, "D": D}
        elif self.loss_type == 'inbatch-negative' and self.pooler_type == 'mean':
            scores_inbatch = Q @ D.permute(1, 0) # (B 2*B) by SentEmbed dot product
            loss = InBatchNegativeCELoss(scores_inbatch)
            return {'score': scores_inbatch, 'loss': loss, "Q": Q, "D": D}
        else:
            return ValueError('Invalid loss type.')


    def pairwise_score(self, Q, D):
        """ Max sim operator for pairwise loss

        1. tokens cos-sim: (B Lq H) X (B H Ld) = (B Lq Ld)
        2. max token-token cos-sim: (B Lq), the last dim indicates max of qd-cos-sim of q
        3. sum by batch: (B 1)
        """
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)
    
    def inbatch_score(self, Q, D):
        Q_prime = Q.view(-1, Q.size(-1)) # (B*Lq H)
        D_prime = D.view(-1, D.size(-1)) # (B*2*Ld H)
        B, Lq, Lh = Q.size(0), Q.size(1), D.size(1)

        # if self.similarity_metric == 'cosine':
        return (Q_prime @ D_prime.permute(1, 0)).view(B, Lq, B*2, Lh).permute(0, 2, 1, 3).max(-1).values.sum(-1) #(B 2B Lq Ld) -> (B 2B Lq) -> (B 2B)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] \
                for d in input_ids.cpu().tolist()]
        mask = torch.tensor(mask, device=self.device).unsqueeze(2).float()
        return mask  # B L 1

    def colbert_pooler(self, tokens_last_hidden, mask=1, keep_dims=True):
        X = self.linear(tokens_last_hidden)
        X = X * mask # for d
        X = F.normalize(X, p=2, dim=2)

        if not keep_dims:  # for d
            X, mask = X.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            X = [d[mask[idx]] for idx, d in enumerate(X)]
        return X
    
    def avg_pooler(self, tokens_last_hidden, attention_mask): # (B Lq H) -> (B H)
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(
                tokens_last_hidden.size()
        ).float()
        tokens_last_hidden = tokens_last_hidden * attention_mask_expanded

        return torch.mean(tokens_last_hidden[:, 4:, :], dim=-2)

    def encoder(self, 
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                source: Optional[str] = 'query'):

        # [TODO]: Reformulate text into query or document inputs
        if not_append_prefix:
            pass

        e = self.bert(
            input_ids=q_input_ids,
            attention_mask=q_attention_mask,
            token_type_ids=q_token_type_ids,
            **kwargs
        )

        if self.pooler_type == 'colbert':
            if source == 'document':
                e_mask = self.mask(e_input_ids)
                E = self.colbert_pooler(e.last_hidden_state, e_mask, True)
            else:
                E = self.colbert_pooler(e.last_hidden_state)
        elif self.pooler_type == 'mean':
            E = self.avg_pooler(e.last_hidden_state. attention_mask)
        else:
            raise ValueError('Invalid pooler_type.')
        
        return E
