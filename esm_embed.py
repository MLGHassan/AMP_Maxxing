"""Frozen ESM-2 (esm2_t6_8M_UR50D, 320-d) per-residue embeddings with an on-disk cache."""
import os
import numpy as np, torch

NAME = "facebook/esm2_t6_8M_UR50D"
TOK_LEN = 52          # 50 residues + <cls> + <eos>


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EsmEmbedder:
    def __init__(self, device=None, cache_path="cache/esm2_t6_8M.pt", name=NAME):
        from transformers import AutoTokenizer, EsmModel
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = EsmModel.from_pretrained(name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.cache_path = cache_path
        self.cache = torch.load(cache_path) if cache_path and os.path.exists(cache_path) else {}
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def embed(self, seqs, batch=64):
        """-> H [N, 52, D] float16 (zero padded), M [N, 52] bool (valid tokens)."""
        todo = [s for s in dict.fromkeys(seqs) if s not in self.cache]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            t = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            h = self.model(**t).last_hidden_state.float().cpu()
            m = t["attention_mask"].cpu().bool()
            for s, hh, mm in zip(chunk, h, m):
                self.cache[s] = hh[mm].half()                    # [L+2, D]
        if todo and self.cache_path and len(todo) > 50:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            torch.save(self.cache, self.cache_path)
        H = torch.zeros(len(seqs), TOK_LEN, self.dim, dtype=torch.float16)
        M = torch.zeros(len(seqs), TOK_LEN, dtype=torch.bool)
        for i, s in enumerate(seqs):
            e = self.cache[s]; H[i, :len(e)] = e; M[i, :len(e)] = True
        return H, M
