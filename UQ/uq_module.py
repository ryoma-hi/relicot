# -*- coding: utf-8 -*-
"""
CoT-UQ (Zhang & Zhang, 2025) を、既存の CoT モジュール上で動かす最小実装。
- Stage 1: Reasoning抽出 → Keywords抽出+重要度付け(1~10)
- Stage 2: AP強化(式5,6) / SE強化(式7,8,9)

依存:
- relicot.CoT.cot_module.CoTGenerator (generate_fn を外から注入)
- HFモデルを用いた対数確率計算 (HFCausalLogProbScorer; 本ファイル内)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re
import math

import torch
import torch.nn.functional as F

from relicot.CoT.cot_module import CoTGenerator
from relicot.CoT.cot_parser import parse_all
from . import prompts as uq_prompts


# -----------------------------
# HF causal LM logprob scorer
# -----------------------------
@dataclass
class LogProbResult:
    total_logprob: float
    avg_logprob: float
    n_tokens: int

class HFCausalLogProbScorer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer
        self.device = getattr(model, "device", None) or next(model.parameters()).device

    @torch.no_grad()
    def seq_logprob(self, prompt: str, continuation: str) -> LogProbResult:
        # continuation が空ならゼロ返し
        cont_ids = self.tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        if cont_ids.numel() == 0:
            return LogProbResult(0.0, 0.0, 0)
        prompt_ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        input_ids  = torch.cat([prompt_ids, cont_ids], dim=1)  # [prompt + cont]
        out = self.model(input_ids=input_ids)
        logits  = out.logits[:, :-1, :]         # next-token
        targets = input_ids[:, 1:]              # shifted
        p_len   = prompt_ids.shape[1]
        start   = max(p_len - 1, 0)             # cont 部分に対応
        cont_logits  = logits[:, start:, :]
        cont_targets = targets[:, start:]
        if cont_targets.shape[1] == 0:
            return LogProbResult(0.0, 0.0, 0)
        log_probs = F.log_softmax(cont_logits, dim=-1)
        tok_lp    = log_probs.gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1)  # (1, T)
        total = float(tok_lp.sum().item())
        n_tok = cont_targets.shape[1]
        return LogProbResult(total, total / max(n_tok, 1), n_tok)

    @torch.no_grad()
    def binary_choice_prob(self, prompt: str, opt_true: str = " True", opt_false: str = " False") -> float:
        """P(True) を2候補の確率正規化で近似。空白付きでトークナイズの安定化。"""
        lp_true  = self.seq_logprob(prompt, opt_true).total_logprob
        lp_false = self.seq_logprob(prompt, opt_false).total_logprob
        m = max(lp_true, lp_false)
        p_true = math.exp(lp_true - m) / (math.exp(lp_true - m) + math.exp(lp_false - m))
        return float(p_true)


# -----------------------------
# Stage 1: prompts & parsing
# -----------------------------


_step_line = re.compile(r"(?mi)^\s*Step\s*(\d+)\s*:\s*(.+)$")
_JA_TOKEN = r"[ぁ-んァ-ヶ一-龥A-Za-z0-9％%．\.\-\/\(\)（）：: ]"

def _clean_kw(s: str) -> str:
    s = s.strip()
    # 長すぎる“説明文”を弾く（例: 20〜30字以上を候補外に）
    if len(s) > 30:
        return ""
    # 日本語主体/記号のみ許容。英語長文は除去
    if not re.match(rf"^{_JA_TOKEN}+$", s):
        return ""
    # 先頭末尾の不要記号を削る
    s = re.sub(r"^[\-\•\・\s]+|[\-\•\・\s]+$", "", s)
    return s

def parse_keywords_block(text: str) -> Dict[int, List[Tuple[str, int]]]:
    out: Dict[int, List[Tuple[str, int]]] = {}
    for m in _step_line.finditer(text):
        idx = int(m.group(1))
        rest = m.group(2).strip()
        if re.search(r"NO\s*ANSWER", rest, re.I):
            out[idx] = []
            continue
        pairs: List[Tuple[str, int]] = []
        # セミコロン区切りで「キーワード (/N/)」を抽出
        for seg in [s.strip() for s in rest.split(";") if s.strip()]:
            m2 = re.match(r"^(.*?)[\s　]*\(\s*/\s*(\d+)\s*/\s*\)\s*$", seg)
            if not m2:
                continue
            kw = _clean_kw(m2.group(1))
            if not kw:
                continue
            sc = int(m2.group(2))
            sc = max(1, min(10, sc))  # 1..10に丸め
            pairs.append((kw, sc))
        # 重複をスコア最大でマージ
        merged: Dict[str, int] = {}
        for k, t in pairs:
            merged[k] = max(merged.get(k, 0), t)
        out[idx] = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:5]  # 各Step上位5まで
    return out

# -----------------------------
# Stage 2: AP 強化 (式5,6)
# -----------------------------
def aggregate_keyword_prob(
    scorer: HFCausalLogProbScorer,
    base_prompt: str,
    keyword: str,
    agg: str = "mean",  # "mean" or "min"
) -> float:
    """
    式(5): p(w^) = Aggr_m P(w_m | p, w_<m)
    ここでは continuation を keyword として seq_logprob を取り、平均/最小で集約。
    """
    res = scorer.seq_logprob(base_prompt, keyword)
    if res.n_tokens == 0:
        return 0.0
    if agg == "mean":
        # 平均確率に近似（対数平均→確率平均は厳密ではないが実装簡便化。スコアの相対比較用途）
        return math.exp(res.avg_logprob)
    elif agg == "min":
        # 近似: 1トークンずつの logprob を取って min(prob) を使いたいが、
        # 実装簡便化のため avg から近似。必要なら拡張。
        return math.exp(res.avg_logprob)  # TODO: tokenwise最小に差し替え
    else:
        return math.exp(res.avg_logprob)

def confidence_ap_weighted(
    scorer: HFCausalLogProbScorer,
    base_prompt: str,
    kw_by_step: Dict[int, List[Tuple[str, int]]],
    agg: str = "mean",
    tau: Optional[int] = None,   # 重要度フィルタ (論文の τ)
    topk_if_few: int = 3,        # KEYKeywords の「最低3個確保」
) -> Dict[str, Any]:
    """
    式(6): c = sum_{i,j} t_ij * p(w_ij^) / sum_{i,j} t_ij
    tau を指定した場合は importance >= tau のみを使用。少なければ上位topk_if_fewで補完。
    """
    # KEYKeywords 的フィルタ
    items: List[Tuple[str, int]] = []
    for _, pairs in kw_by_step.items():
        for kw, t in pairs:
            items.append((kw, t))
    if tau is not None:
        filt = [(k, t) for (k, t) in items if t >= tau]
        if len(filt) < min(topk_if_few, len(items)):
            # 重要度降順で上位補完
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
            filt_set = set(filt)
            for it in items_sorted:
                if it not in filt_set:
                    filt.append(it)
                if len(filt) >= min(3, len(items_sorted)):
                    break
        items = filt

    if not items:
        return {"confidence": 0.0, "used": [], "denom": 0.0}

    num = 0.0
    den = 0.0
    used = []
    for kw, t in items:
        p_kw = aggregate_keyword_prob(scorer, base_prompt, kw, agg=agg)
        num += t * p_kw
        den += t
        used.append({"keyword": kw, "t": t, "p": p_kw})
    conf = num / den if den > 0 else 0.0
    return {"confidence": conf, "used": used, "denom": den}


# -----------------------------
# Stage 2: SE 強化 (式7,8,9)
# -----------------------------
def build_se_context(
    question: str,
    answer: str,
    steps_text: str,
    kw_by_step: Dict[int, List[Tuple[str, int]]],
    mode: str = "ALLSteps",   # "ALLSteps" | "ALLKeywords" | "KEYStep" | "KEYKeywords"
    tau: int = 7              # KEYKeywords 用のしきい値（論文では中程度が良い傾向）
) -> Tuple[str, Dict[str, Any]]:
    """
    SE の追加情報を組み立てる（Appendix B.2 のテンプレに準拠）。
    戻り値: (追加情報テキスト, メタ)
    """
    meta: Dict[str, Any] = {}
    if mode == "ALLSteps":
        info = f"A step-by-step reasoning to the possible answer:\n{steps_text}"
    elif mode == "ALLKeywords":
        lines = []
        for i in sorted(kw_by_step.keys()):
            pairs = kw_by_step[i]
            if pairs:
                lines.append(f"Step {i}: " + "; ".join([f"{k} (/{t}/)" for k,t in pairs]))
            else:
                lines.append(f"Step {i}: NO ANSWER")
        info = "Keywords during reasoning to the possible answer:\n" + "\n".join(lines)
    elif mode == "KEYStep":
        # 式(7): 各ステップの平均重要度の最大
        best_i = None
        best_score = -1.0
        for i, pairs in kw_by_step.items():
            if not pairs:
                continue
            avg = sum(t for _, t in pairs) / max(len(pairs), 1)
            if avg > best_score:
                best_score = avg
                best_i = i
        meta["key_step"] = best_i
        if best_i is None:
            info = "The most critical step is not available."
        else:
            pairs = kw_by_step[best_i]
            info = f"The most critical step in reasoning is: Step {best_i}\n" + "; ".join([f"{k} (/{t}/)" for k,t in pairs])
    elif mode == "KEYKeywords":
        # 式(8): 重要度 τ 以上のキーワードのみ。少なければ上位で補完（最低3）
        all_items = []
        for i, pairs in kw_by_step.items():
            for k,t in pairs:
                all_items.append((k,t))
        chosen = [(k,t) for (k,t) in all_items if t >= tau]
        if len(chosen) < min(3, len(all_items)):
            all_items_sorted = sorted(all_items, key=lambda x: x[1], reverse=True)
            for it in all_items_sorted:
                if it not in chosen:
                    chosen.append(it)
                if len(chosen) >= min(3, len(all_items_sorted)):
                    break
        meta["n_key_keywords"] = len(chosen)
        info = "The most critical keywords during reasoning are:\n" + "; ".join([f"{k} (/{t}/)" for k,t in chosen]) if chosen else "No critical keywords found."
    else:
        info = ""
    return info, meta


def p_true_with_context(
    scorer: HFCausalLogProbScorer, question: str, answer: str, added_info: str
) -> float:
    """True/False自己評価（テンプレは外部ファイル）。"""
    prompt = uq_prompts.render(
        "se_true_false_probe_ja",
        vars={"question": question, "answer": answer, "added_info": added_info},
    )
    return scorer.binary_choice_prob(prompt, opt_true=" True", opt_false=" False")


# -----------------------------
# High-level API
# -----------------------------
@dataclass
class CoTUQOutput:
    problem: str
    steps: List[str]
    answer: str
    kw_by_step: Dict[int, List[Tuple[str, int]]]
    ap_confidence: Optional[float]
    se_confidences: Dict[str, float]
    raw: str
    meta: Dict[str, Any]

def run_cotuq(
    gen: CoTGenerator,
    question: str,
    *,
    tokenizer=None,
    # Stage 1
    template_reason: str = "cot_default_ja",
    template_kw: str = "kw_extract_ja",
    template_se_probe: str = "se_true_false_probe_ja",
    # Stage 2 (AP)
    scorer: Optional[HFCausalLogProbScorer] = None,
    ap_use: bool = True,
    ap_agg: str = "mean",
    ap_tau: Optional[int] = None,
    # Stage 2 (SE)
    se_modes: List[str] = ("ALLSteps","ALLKeywords","KEYStep","KEYKeywords"),
    se_tau: int = 7,
) -> CoTUQOutput:
    """
    論文の CoT-UQ を1クエリで実行。AP/SEは片方だけでも可。
    """
    print(f"[START] Question: {question}")

    # ---- Stage 1: reasoning ----
    print("[Stage1] Running CoT reasoning...")
    out = gen.run(problem=question, template=template_reason, tokenizer=tokenizer)
    steps = out["steps"]
    answer = out["answer"] or ""
    steps_text = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(steps)])
    print("[Stage1] Done. Steps:")
    for i, s in enumerate(steps, 1):
        print(f"  Step {i}: {s}")
    print(f"[Stage1] Final Answer: {answer}")

    # ---- Stage 1: keywords+scores
    print("[Stage1] Extracting keywords+scores...")
    kw_prompt = uq_prompts.render(
        template_kw,
        vars={"question": question, "multistep": steps_text + f"\nFinal Answer: {answer}"}
    )
    kw_text = gen.generate_fn(
        kw_prompt,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        stop=["\n\nQuestion:", "\nQ:", "\n問題:"],
    )
    kw_by_step = parse_keywords_block(kw_text)
    print("[Stage1] Keywords extracted:")
    for step_id, kws in kw_by_step.items():
        print(f"  Step {step_id}: {kws}")

    # ---- Stage 2: AP
    ap_conf = None
    ap_detail = None
    if ap_use and scorer is not None:
        print("[Stage2] Computing AP (answer probability weighted by keywords)...")
        ap_out = confidence_ap_weighted(
            scorer=scorer,
            base_prompt="Question: " + question + "\n",
            kw_by_step=kw_by_step,
            agg=ap_agg,
            tau=ap_tau,
        )
        ap_conf = ap_out["confidence"]
        ap_detail = ap_out
        print(f"[Stage2] AP Confidence: {ap_conf:.4f}")

    # ---- Stage 2: SE
    se_confs: Dict[str, float] = {}
    if scorer is not None:
        print("[Stage2] Computing SE (self-evaluation) for each mode...")
        for mode in se_modes:
            print(f"  > Mode {mode} ...", end=" ")
            added, meta_mode = build_se_context(
                question=question,
                answer=answer,
                steps_text=steps_text,
                kw_by_step=kw_by_step,
                mode=mode,
                tau=se_tau,
            )
            # テンプレ名を使い回す場合は p_true_with_context 内はそのままでOK
            ptrue = p_true_with_context(scorer, question, answer, added)
            se_confs[mode] = ptrue
            print(f"done. P(True)={ptrue:.4f}")

    print("[END] Finished CoT-UQ.\n")

    return CoTUQOutput(
        problem=question,
        steps=steps,
        answer=answer,
        kw_by_step=kw_by_step,
        ap_confidence=ap_conf,
        se_confidences=se_confs,
        raw=out["raw"],
        meta={
            "prompt_reason": out["prompt"],
            "kw_raw": kw_text,
            "ap_detail": ap_detail,
        },
    )
