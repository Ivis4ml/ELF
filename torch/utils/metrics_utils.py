"""BLEU / ROUGE + GPT-2-based generative perplexity."""

import math
import statistics
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from utils.logging_utils import log_for_0


def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    import sacrebleu
    return sacrebleu.corpus_bleu(
        hypotheses, [references], lowercase=True, use_effective_order=True,
    ).score


def compute_rouge(hypotheses: List[str], references: List[str], return_std: bool = False):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for hyp, ref in zip(hypotheses, references):
        s = scorer.score(ref, hyp)
        r1.append(s["rouge1"].fmeasure * 100)
        r2.append(s["rouge2"].fmeasure * 100)
        rL.append(s["rougeL"].fmeasure * 100)
    def _mss(v):
        n = len(v)
        m = sum(v) / max(n, 1)
        s = statistics.pstdev(v) if n > 1 else 0.0
        e = s / math.sqrt(n) if n > 1 else 0.0
        return m, s, e
    m1, s1, e1 = _mss(r1); m2, s2, e2 = _mss(r2); mL, sL, eL = _mss(rL)
    means = {"rouge1": m1, "rouge2": m2, "rougeL": mL}
    if not return_std:
        return means
    return means, {"rouge1_std": s1, "rouge2_std": s2, "rougeL_std": sL,
                   "rouge1_sem": e1, "rouge2_sem": e2, "rougeL_sem": eL}


# -------------- PPL --------------

class PerplexityEvaluator:
    """Compute generative PPL under a frozen causal LM (default: gpt2-large)."""

    def __init__(self, model_name: str = "gpt2-large", batch_size: int = 8,
                 context_size: int = 1024, device: Optional[torch.device] = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.model_name = model_name
        self.batch_size = batch_size
        self.context_size = context_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        use_fast = "mt5" not in model_name.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        log_for_0(f"Loading PPL model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def evaluate(self, text_samples: List[str], max_length: int) -> Dict:
        tok = self.tokenizer(
            text_samples, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length, return_attention_mask=True,
        )
        ids = tok["input_ids"].to(self.device)
        attn = tok["attention_mask"].to(self.device)
        eos_id = self.tokenizer.eos_token_id

        nll_sum, tok_count = 0.0, 0.0
        per_sample_nll = np.zeros(ids.shape[0], dtype=np.float64)
        per_sample_tok = np.zeros(ids.shape[0], dtype=np.float64)

        B_total = ids.shape[0]
        bs = max(1, self.batch_size)
        for i in tqdm(range(0, B_total, bs), desc="PPL"):
            bi = ids[i:i + bs]
            ba = attn[i:i + bs]
            logits = self.model(bi, attention_mask=ba).logits.float()
            targets = bi[:, 1:]
            log_pred = logits[:, :-1]
            log_probs = torch.log_softmax(log_pred, dim=-1)
            tgt_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            nlls = -tgt_lp

            is_eos = (bi == eos_id)
            first_eos = (torch.cumsum(is_eos.long(), dim=-1) == 1)
            non_pad = (bi != eos_id)
            valid = first_eos[:, 1:] | non_pad[:, 1:]

            nlls = nlls * valid.float()
            per_sample_nll[i:i + bs] = nlls.sum(-1).cpu().numpy()
            per_sample_tok[i:i + bs] = valid.float().sum(-1).cpu().numpy()
            nll_sum += float(nlls.sum().item())
            tok_count += float(valid.float().sum().item())

        with np.errstate(divide="ignore", invalid="ignore"):
            ppl_per = np.exp(per_sample_nll / per_sample_tok)
        ppl_per = np.where(per_sample_tok > 0, ppl_per, np.nan).tolist()

        per_sample_entropy = []
        ids_np = ids.cpu().numpy()
        attn_np = attn.cpu().numpy()
        for j in range(ids.shape[0]):
            valid_len = int(attn_np[j].sum())
            uniq, counts = np.unique(ids_np[j, :valid_len], return_counts=True)
            probs = counts.astype(np.float32) / counts.sum()
            per_sample_entropy.append(float(-np.sum(probs * np.log(probs + 1e-10))))

        return {
            "ppl": math.exp(nll_sum / max(tok_count, 1.0)),
            "per_sample_ppl": ppl_per,
            "mean_entropy": sum(per_sample_entropy) / max(len(per_sample_entropy), 1),
        }
