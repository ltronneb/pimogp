import torch
from torch import nn

from pimogp.models.transformer_vae import TransformerEncoder, TransformerDecoder, make_causal_mask


class SelfiesEfficacyAutoencoder(nn.Module):
    """
    Small, deterministic (non-generative) SELFIES autoencoder: encode a drug to
    a bottleneck z, decode back to SELFIES logits, and predict KPL1 monotherapy
    efficacy off the same z. No reparameterization/KL term -- z is a plain
    compressed representation, not a sampled latent, so the embedding is
    shaped by reconstruction + efficacy, not by a generative prior.
    """
    def __init__(self, vocab_size, embed_size=64, num_layers=2, num_heads=4,
                 hidden_dim=128, z_dim=32, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, embed_size, num_layers, num_heads, hidden_dim, max_seq_len, dropout)
        self.decoder = TransformerDecoder(vocab_size, embed_size, num_layers, num_heads, hidden_dim, max_seq_len, dropout)
        self.fc_z = nn.Linear(embed_size, z_dim)
        self.z2mem = nn.Linear(z_dim, embed_size)
        self.head = nn.Linear(z_dim, 1)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_z(h)

    def decode(self, z, x, tgt_mask=None, tgt_key_padding_mask=None):
        mem = self.z2mem(z).unsqueeze(1).repeat(1, x.size(1), 1)
        return self.decoder(x, mem, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

    def forward(self, x, pad_idx):
        z = self.encode(x)
        efficacy = self.head(z).squeeze(-1)

        x_input = x[:, :-1]
        x_target = x[:, 1:]

        causal_mask = make_causal_mask(x_input.size(1), x.device)
        pad_mask = (x_input == pad_idx)

        logits = self.decode(z, x_input, tgt_mask=causal_mask, tgt_key_padding_mask=pad_mask)
        return logits, z, efficacy, x_target


class SelfiesCurveAutoencoder(nn.Module):
    """
    Same as SelfiesEfficacyAutoencoder, but the auxiliary head predicts a
    drug's full monotherapy dose-response curve (n_conc_levels points, one
    per ONeil concentration level) instead of a single efficacy scalar --
    real ONeil KPL1 potency data (see EFFICACY_EMBEDDING_EXPERIMENTS.md),
    not the drugs.csv efficacy column, which was found not to correlate
    with it at all. Sigmoid-bounded output since viability in [0,1].
    """
    def __init__(self, vocab_size, embed_size=64, num_layers=2, num_heads=4,
                 hidden_dim=128, z_dim=32, max_seq_len=100, dropout=0.1, n_conc_levels=10):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, embed_size, num_layers, num_heads, hidden_dim, max_seq_len, dropout)
        self.decoder = TransformerDecoder(vocab_size, embed_size, num_layers, num_heads, hidden_dim, max_seq_len, dropout)
        self.fc_z = nn.Linear(embed_size, z_dim)
        self.z2mem = nn.Linear(z_dim, embed_size)
        self.head = nn.Linear(z_dim, n_conc_levels)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_z(h)

    def decode(self, z, x, tgt_mask=None, tgt_key_padding_mask=None):
        mem = self.z2mem(z).unsqueeze(1).repeat(1, x.size(1), 1)
        return self.decoder(x, mem, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

    def forward(self, x, pad_idx):
        z = self.encode(x)
        curve = torch.sigmoid(self.head(z))   # (batch, n_conc_levels)

        x_input = x[:, :-1]
        x_target = x[:, 1:]

        causal_mask = make_causal_mask(x_input.size(1), x.device)
        pad_mask = (x_input == pad_idx)

        logits = self.decode(z, x_input, tgt_mask=causal_mask, tgt_key_padding_mask=pad_mask)
        return logits, z, curve, x_target
