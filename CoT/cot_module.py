# -*- coding: utf-8 -*-
"""
Core CoT module.
CoTGenerator
├─ run()               # 基本のCoT実行
├─ run_self_consistency() # 複数サンプルから多数決
├─ run_json()          # JSON構造で返させる
└─ run_for_uq()        # UQ(不確実性定量)用のフォーマットで返す


Usage pattern
------------
from relicot.CoT.cot_module import CoTGenerator
from relicot.CoT.cot_prompts import render
from relicot.CoT import cot_prompts

# 1) Provide a generation function (HF example):
def hf_generate(prompt: str, max_new_tokens=256, temperature=0.7, top_p=0.95, stop=None):
    # pseudo-implementation; integrate your HF model here
    # outputs should be plain text
    # - respect 'stop' if provided (truncate at first occurrence)
    return model_generate_text_somehow(prompt, max_new_tokens, temperature, top_p, stop)

# 2) Build the CoT prompt:
p = cot_prompts.render("cot_default_ja", problem="犬は色を識別できますか？")

# 3) Run CoT:
gen = CoTGenerator(generate_fn=hf_generate)
result = gen.run(problem="犬は色を識別できますか？", template="cot_default_ja")

print(result["steps"])
print(result["answer"])

Notes
-----
- `generate_fn` must be a callable: (prompt: str, **decode_kwargs) -> str
- Self-consistency: `run_self_consistency(...)` does k samples and majority-votes final answer.
"""
from __future__ import annotations
from typing import Callable, Dict, Any, List, Optional
# 1) 型エイリアスを追加
GenerateFn = Callable[..., str]
RenderFn   = Callable[..., str]   # ← 追加

from dataclasses import dataclass

from .cot_prompts import render as render_prompt
from .cot_parser import parse_all
from .cot_utils import (
    safe_json_loads,
    extract_first_json_block,
    majority_vote,
    truncate_tokens,
)

GenerateFn = Callable[..., str]

@dataclass
class CoTOutput:
    steps: List[str]
    answer: Optional[str]
    raw: str
    meta: Dict[str, Any]

class CoTGenerator:
    # 2) __init__ に render_fn を注入できるように
    def __init__(self, generate_fn: GenerateFn, max_input_tokens: int = 4096,
                 render_fn: RenderFn = render_prompt):   # ← 追加
        self.generate_fn = generate_fn
        self.max_input_tokens = max_input_tokens
        self.render_fn = render_fn                        # ← 追加

    # 3) 生プロンプトをそのまま叩くユーティリティ（任意）
    def generate_raw(self, prompt: str,
                     decode_kwargs: Optional[Dict[str, Any]] = None,
                     tokenizer=None) -> str:
        decode_kwargs = decode_kwargs or {"max_new_tokens": 256, "temperature": 0.7}
        prompt = truncate_tokens(prompt, tokenizer=tokenizer, max_tokens=self.max_input_tokens)
        return self.generate_fn(prompt, **decode_kwargs)

    # 4) run 系に prompt_override を追加して、render_fn 経由に
    def run(self, problem: str, template: str = "cot_default_ja", instruction: str = "",
            decode_kwargs: Optional[Dict[str, Any]] = None, tokenizer=None,
            prompt_override: Optional[str] = None) -> Dict[str, Any]:
        decode_kwargs = decode_kwargs or {"max_new_tokens": 256, "temperature": 0.7}
        prompt = (prompt_override or
                  self.render_fn(template, problem=problem, instruction=instruction))
        prompt = truncate_tokens(prompt, tokenizer=tokenizer, max_tokens=self.max_input_tokens)
        text = self.generate_fn(prompt, **decode_kwargs)
        parsed = parse_all(text)
        return {"steps": parsed.steps, "answer": parsed.answer, "raw": parsed.raw,
                "prompt": prompt, "decode_kwargs": decode_kwargs}

    def run_self_consistency(self, problem: str, template: str = "cot_sc_ja", k: int = 8,
                             instruction: str = "", decode_kwargs: Optional[Dict[str, Any]] = None,
                             tokenizer=None, prompt_override: Optional[str] = None) -> Dict[str, Any]:
        decode_kwargs = decode_kwargs or {"max_new_tokens": 192, "temperature": 0.9, "top_p": 0.95}
        prompt = (prompt_override or
                  self.render_fn(template, problem=problem, instruction=instruction))
        prompt = truncate_tokens(prompt, tokenizer=tokenizer, max_tokens=self.max_input_tokens)
        # （以下は元のまま）
        answers, raws, steps_all = [], [], []
        for i in range(k):
            text = self.generate_fn(prompt, **decode_kwargs)
            pr = parse_all(text)
            answers.append(pr.answer); steps_all.append(pr.steps); raws.append(pr.raw)
        voted = majority_vote([a for a in answers if a is not None])
        return {"answer_majority": voted, "answers": answers, "steps_list": steps_all,
                "raw_list": raws, "prompt": prompt, "decode_kwargs": decode_kwargs}

    def run_json(self, problem: str, template: str = "cot_json_ja",
                 decode_kwargs: Optional[Dict[str, Any]] = None, tokenizer=None,
                 strict: bool = False, prompt_override: Optional[str] = None) -> Dict[str, Any]:
        decode_kwargs = decode_kwargs or {"max_new_tokens": 256, "temperature": 0.3}
        prompt = (prompt_override or self.render_fn(template, problem=problem))
        prompt = truncate_tokens(prompt, tokenizer=tokenizer, max_tokens=self.max_input_tokens)
        text = self.generate_fn(prompt, **decode_kwargs)
        block = extract_first_json_block(text)
        data = safe_json_loads(block or "")
        if data is None:
            if strict: return {}
            pr = parse_all(text)
            return {"steps": pr.steps, "answer": pr.answer, "raw": pr.raw,
                    "prompt": prompt, "decode_kwargs": decode_kwargs,
                    "note": "fallback from invalid JSON"}
        return {"steps": data.get("steps", []), "answer": data.get("answer"),
                "raw": text, "prompt": prompt, "decode_kwargs": decode_kwargs}

    # -------- Adapter for CoT-UQ --------
    def run_for_uq(
        self,
        problem: str,
        template: str = "cot_default_ja",
        instruction: str = "",
        decode_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer=None,
    ) -> CoTOutput:
        """
        Produce CoT result tailored for UQ modules:
        - returns steps[], answer, raw, meta (prompt, decode_kwargs)
        CoT-UQ（不確実性定量化）モジュールに渡しやすい形式を返す。
        """
        out = self.run(
            problem=problem,
            template=template,
            instruction=instruction,
            decode_kwargs=decode_kwargs,
            tokenizer=tokenizer,
        )
        return CoTOutput(
            steps=out["steps"],
            answer=out["answer"],
            raw=out["raw"],
            meta={"prompt": out["prompt"], "decode_kwargs": out["decode_kwargs"]},
        )
