import torch
import torch.nn as nn
import math

trained_embeddings = torch.load('embed.pth')

class PositionalEncoding(nn.Module):
    """Pytorch positionalEncoding implementation from https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Args:
            x : Tensor of shape (batches, seq_len, embedding_dim)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EmmbendingModel(nn.Module):
    """Model for creating meaningful embeddings for the champions in the game"""
    def __init__(self, num_champs: int, embed_dim: int, num_heads: int, d_prob=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.context_dim = 24
        self.total_dim = embed_dim + self.context_dim

        self.embed = nn.Embedding(num_champs, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.total_dim, nhead=num_heads,
                                                        dropout=d_prob, dim_feedforward=self.total_dim * 4,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.total_dim * 10, 1)
        self.batch = nn.BatchNorm1d(10)



    def forward(self, x, context):
        x = self.embed(x)  # Shape: (batch_size, seq_len, embed_dim)
        x = self.pos_encoder(x)  # Add positional encoding
        context = self.batch(context)
        # Concatenate context to embeddings
        x = torch.cat((x, context), dim=-1)  # Shape: (batch_size, seq_len, total_dim)

        x = self.transformer_encoder(x)  # Pass through the transformer encoder
        x = self.flatten(x)  # Flatten the output
        x = self.linear(x)  # Final linear layer to produce output

        return x


    def cos_similarity(self, c1, c2):
        # use the predefined self.cos similarity object
        return self.cos(self.embed(c1), self.embed(c2))

    def get_embbeding(self,e):
      return self.embed(e)


class PredictingModel(nn.Module):
    """Models the outcome of a match of the form [p1_champ, p2_champ, ..., p10_champ] using
    a multi-layer Transformer Encoder architecture."""

    def __init__(self, num_champs: int, embed_dim: int, num_heads: int, d_prob=0.1):
        super(PredictingModel, self).__init__()

        self.embed_dim = embed_dim
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.embed = nn.Embedding(num_champs, embed_dim, _weight=trained_embeddings, _freeze=True)
        self.pos_encoder = PositionalEncoding(embed_dim)
        # Transformer Encoder for champions
        self.encode_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                       dropout=d_prob, dim_feedforward=embed_dim,
                                                       batch_first=True)
        self.team_comp_encoder = nn.TransformerEncoder(self.encode_layer, 6)

        # Output
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.embed_dim * 10, self.embed_dim * 10)
        self.linear2 = nn.Linear(self.embed_dim * 10, 1)
        self.Relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(self.embed_dim * 10)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (#batches, 10).

        Returns:
            probes: Tensor of shape (#batches, 1)."""

        match_emb = self.embed(x)
        match_emb = self.pos_encoder(match_emb)
        encoded = self.team_comp_encoder(match_emb)
        flattened = self.flatten(encoded)
        probs = self.linear(flattened)
        probs = self.batch(probs)
        probs = self.Relu(probs)
        probs = self.linear2(probs)

        return probs

