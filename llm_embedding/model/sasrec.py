import torch
import torch.nn as nn
import copy
from model._abstract_model import SequentialRecModel
from model._modules import TransformerEncoder, LayerNorm
from model.embedding_test import convert_embedding_vector
import model.embedding_test as model_emb

class SASRecModel(SequentialRecModel):
    def __init__(self, args):
        super(SASRecModel, self).__init__(args)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        # Define item_embeddings layer
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.item_encoder = TransformerEncoder(args)
        self.apply(self.init_weights)
        
        # 추가
        self.args = args
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)

        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_out = self.forward(input_ids)
        seq_out = seq_out[:, -1, :]
        pos_ids, neg_ids = answers, neg_answers

        # Ensure pos_ids and neg_ids are tensors
        if not isinstance(pos_ids, torch.Tensor):
            pos_ids = torch.tensor(pos_ids, device=seq_out.device)
        if not isinstance(neg_ids, torch.Tensor):
            neg_ids = torch.tensor(neg_ids, device=seq_out.device)

        pos_emb = convert_embedding_vector(self.hidden_size, self.batch_size, pos_ids.tolist(), model_emb.embedding_vec_dict, self.args.max_seq_length)
        neg_emb = convert_embedding_vector(self.hidden_size, self.batch_size, neg_ids.tolist(), model_emb.embedding_vec_dict, self.args.max_seq_length)

        # [batch hidden_size]
        seq_emb = seq_out  # [batch, hidden_size]

        # Reshape pos_emb and neg_emb to [batch, hidden_size]
        pos_emb = pos_emb.view(seq_emb.size(0), -1, self.hidden_size)
        neg_emb = neg_emb.view(seq_emb.size(0), -1, self.hidden_size)

        # Flatten pos_emb and neg_emb
        pos_emb = pos_emb[:, -1, :]  # Take the last embedding in the sequence
        neg_emb = neg_emb[:, -1, :]  # Take the last embedding in the sequence

        assert pos_emb.size(1) == self.hidden_size, f"pos_emb size mismatch: {pos_emb.size(1)} vs {self.hidden_size}"
        assert neg_emb.size(1) == self.hidden_size, f"neg_emb size mismatch: {neg_emb.size(1)} vs {self.hidden_size}"

        pos_logits = torch.sum(pos_emb * seq_emb, -1)  # [batch]
        neg_logits = torch.sum(neg_emb * seq_emb, -1)  # [batch]

        pos_labels = torch.ones(pos_logits.shape, device=seq_out.device)
        neg_labels = torch.zeros(neg_logits.shape, device=seq_out.device)

        indices = (pos_ids != 0).nonzero(as_tuple=True)[0]  # Get indices where pos_ids is not zero
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])

        return loss
