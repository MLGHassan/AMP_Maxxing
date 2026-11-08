"""Model ladder.  rung 2: CNN-BiLSTM (Veltri-style) | rung 3: ESM mean-pool | rung 4: ESM attention-pool.
(rung 1, descriptor LR/RF, lives in baselines.py)"""
import numpy as np, torch, torch.nn as nn
from common import AA2ID, MAX_LEN


class MeanHead(nn.Module):
    def __init__(self, d=320, h=64, p=0.2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, H, M):
        m = M.unsqueeze(-1).float()
        return self.mlp((H * m).sum(1) / m.sum(1).clamp(min=1))


class AttnHead1(nn.Module):
    """Single-head additive attention pooling (kept as an ablation of MultiAttnHead)."""
    def __init__(self, d=320, h=64, p=0.2):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1))
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, H, M):
        s = self.att(H).masked_fill(~M.unsqueeze(-1), -1e4)
        w = torch.softmax(s, dim=1)
        return self.mlp((w * H).sum(1))


class MultiAttnHead(nn.Module):
    """Rung 4 of the proposal: MULTI-HEAD attention pooling instead of mean pooling.

    `heads` independent attention distributions are learned over the residue
    positions; each pools the frozen ESM-2 states into its own summary vector and
    the concatenation is classified.  One head can only express a single weighted
    average, which is what makes it a strict generalisation of rung 3.
    """
    def __init__(self, d=320, h=64, p=0.2, heads=4):
        super().__init__()
        self.heads = heads
        self.att = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, heads))
        self.mlp = nn.Sequential(nn.Linear(d * heads, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, H, M):
        s = self.att(H).masked_fill(~M.unsqueeze(-1), -1e4)      # [B, T, heads]
        w = torch.softmax(s, dim=1)
        pooled = torch.einsum("btk,btd->bkd", w, H)              # [B, heads, D]
        return self.mlp(pooled.flatten(1))


class CNNBiLSTM(nn.Module):
    """Embedding -> Conv1D -> MaxPool -> BiLSTM -> sigmoid (after Veltri et al. 2018)."""
    def __init__(self, emb=128, filters=64, k=16, pool=5, hid=100):
        super().__init__()
        self.emb = nn.Embedding(len(AA2ID) + 1, emb, padding_idx=0)
        self.conv = nn.Conv1d(emb, filters, k, padding=k // 2)
        self.pool = nn.MaxPool1d(pool, pool)
        self.lstm = nn.LSTM(filters, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * hid, 1))

    def forward(self, X, M=None):
        z = self.pool(torch.relu(self.conv(self.emb(X).transpose(1, 2)))).transpose(1, 2)
        _, (h, _) = self.lstm(z)
        return self.fc(torch.cat([h[0], h[1]], dim=1))


def encode_ids(seqs):
    X = torch.zeros(len(seqs), MAX_LEN, dtype=torch.long)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = torch.tensor([AA2ID[a] for a in s])
    return X


HEADS = {"esm_attn": MultiAttnHead,     # rung 4: multi-head attention pooling
         "esm_attn1h": AttnHead1,       #   ablation: the same with one head
         "esm_mean": MeanHead,          # rung 3: mean pooling
         "cnn_lstm": CNNBiLSTM}         # rung 2: Veltri-style CNN-BiLSTM


class Scorer:
    """Callable wrapper: list[str] -> np.ndarray of P(AMP)."""
    def __init__(self, kind, head, embedder=None, threshold=0.5, device="cpu"):
        self.kind, self.head, self.emb, self.threshold, self.device = kind, head.to(device).eval(), embedder, threshold, device

    @torch.no_grad()
    def predict(self, seqs, batch=128):
        out = []
        for i in range(0, len(seqs), batch):
            chunk = list(seqs[i:i + batch])
            if self.kind == "cnn_lstm":
                logit = self.head(encode_ids(chunk).to(self.device))
            else:
                H, M = self.emb.embed(chunk)
                logit = self.head(H.float().to(self.device), M.to(self.device))
            out.append(torch.sigmoid(logit).squeeze(-1).cpu().numpy())
        return np.concatenate(out)


def load_scorer(path, embedder=None, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    head = HEADS[ck["kind"]]()
    head.load_state_dict(ck["state_dict"])
    return Scorer(ck["kind"], head, embedder, ck["threshold"], device)
