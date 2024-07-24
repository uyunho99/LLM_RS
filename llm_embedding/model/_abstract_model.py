import torch
import torch.nn as nn
from model._modules import LayerNorm
from torch.nn.init import xavier_uniform_
from model.embedding_test import convert_embedding_vector
import model.embedding_test as model_emb

class SequentialRecModel(nn.Module):
    def __init__(self, args):
        super(SequentialRecModel, self).__init__()
        self.args = args
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.embedding_vec_dict = args.embedding_vec_dict

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand(sequence.size(0), seq_length)
        
        # item_embeddings: text 데이터의 임베딩 벡터로 활용 ([batch_size, seq_length, hidden_size])
        item_embeddings = convert_embedding_vector(self.hidden_size, self.batch_size, sequence, model_emb.embedding_vec_dict, self.args.max_seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        
        # item_embeddings의 길이가 position_embeddings보다 짧을 경우 padding을 추가
        if position_embeddings.size(1) < item_embeddings.size(1):
            padding = torch.zeros(position_embeddings.size(0), item_embeddings.size(1) - position_embeddings.size(1), self.hidden_size, device=sequence.device)
            position_embeddings = torch.cat([position_embeddings, padding], dim=1)
        
        # item_embeddings의 길이가 position_embeddings보다 길 경우 잘라냄
        elif position_embeddings.size(1) > item_embeddings.size(1):
            position_embeddings = position_embeddings[:, :item_embeddings.size(1)]
        
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def init_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_bi_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-head attention."""

        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64

        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""

        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(self, input_ids, all_sequence_output=False):
        pass

    def predict(self, input_ids, user_ids, all_sequence_output=False):
        return self.forward(input_ids, user_ids, all_sequence_output)

    def calculate_loss(self, input_ids, answers):
        pass
