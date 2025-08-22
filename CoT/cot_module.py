# -*- coding: utf-8 -*-
"""
Core CoT module (clean ver.).
- 既存の関数名/引数/返り値(dict)は維持
- 内部では dataclass を使い型を明確化
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, Any, List, Optional

from .cot_prompts import render as render_prompt_default
from .cot_parser import parse_all
from .cot_utils import safe_json_loads, extract_first_json_block, majority_vote, truncate_tokens

# --------- Type aliases ----------
GenerateFn = Callable[..., str]   # (prompt: str, **decode_kwargs) -> str
RenderFn   = Callable[..., str]   # (template: str, **vars) -> str

# --------- Datamodels ----------
@dataclass
class CoTOutput:
    steps: List[str]
    answer: Optional[str]
    raw: str
    prompt: str
    decode_kwargs: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SelfConsistencyOutput:
    answer_majority: Optional[str]
    answers: List[Optional[str]]
    steps_list: List[List[str]]
    raw_list: List[str]
    prompt: str
    decode_kwargs: Dict[str, Any]
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# --------- Defaults ----------
_DEFAULT_DECODE      = {"max_new_tokens": 256, "temperature": 0.7}
_SC_DECODE           = {"max_new_tokens": 192, "temperature": 0.9, "top_p": 0.95}
_JSON_MODE_DECODE    = {"max_new_tokens": 256, "temperature": 0.3}

def _merge_decode(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {**base, **(override or {})}

class CoTGenerator:
    def __init__(self,
                 generate_fn: GenerateFn,
                 *,
                 max_input_tokens: int = 4096,
                 render_fn: RenderFn = render_prompt_default):
        self.generate_fn = generate_fn
        self.max_input_tokens = max_input_tokens
        self.render_fn = render_fn

    # ---- helpers ---------------------------------------------------------
	def _build_prompt(self, template: str, problem: str, instruction: str = "",
            prompt_override: Optional[str] = None,
            render_vars: Optional[Dict[str, Any]] = None) -> str:
        if prompt_override:
            return prompt_override
        v = {"problem": problem, "instruction": instruction}
        if render_vars: v.update(render_vars)
        return self.render_fn(template, **v)

    def _generate(self, prompt: str, decode_kwargs: Dict[str, Any], tokenizer=None) -> str:
        prompt = truncate_tokens(prompt, tokenizer=tokenizer, max_tokens=self.max_input_tokens)
        return self.generate_fn(prompt, **decode_kwargs)

    # ---- public APIs (互換維持) ------------------------------------------
    def generate_raw(self, prompt: str,
                     decode_kwargs: Optional[Dict[str, Any]] = None,
                     tokenizer=None) -> str:
        dk = _merge_decode(_DEFAULT_DECODE, decode_kwargs)
        return self._generate(prompt, dk, tokenizer=tokenizer)

    def run(self, problem: str, template: str = "cot_default_ja", instruction: str = "",
            decode_kwargs: Optional[Dict[str, Any]] = None, tokenizer=None,
            prompt_override: Optional[str] = None,
            render_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        dk = _merge_decode(_DEFAULT_DECODE, decode_kwargs)
        prompt = self._build_prompt(template, problem, instruction, prompt_override, render_vars)
        text = self._generate(prompt, dk, tokenizer=tokenizer)
        parsed = parse_all(text)
        out = CoTOutput(steps=parsed.steps, answer=parsed.answer, raw=parsed.raw,
                        prompt=prompt, decode_kwargs=dk)
        return out.to_dict()

    def run_self_consistency(self, problem: str, template: str = "cot_sc_ja", k: int = 8,
                             instruction: str = "", decode_kwargs: Optional[Dict[str, Any]] = None,
                             tokenizer=None, prompt_override: Optional[str] = None) -> Dict[str, Any]:
        dk = _merge_decode(_SC_DECODE, decode_kwargs)
        prompt = self._build_prompt(template, problem, instruction, prompt_override)
        raws, answers, steps_list = [], [], []
        for _ in range(k):
            text = self._generate(prompt, dk, tokenizer=tokenizer)
            pr = parse_all(text)
            raws.append(pr.raw); answers.append(pr.answer); steps_list.append(pr.steps)
        voted = majority_vote([a for a in answers if a is not None])
        out = SelfConsistencyOutput(
            answer_majority=voted, answers=answers, steps_list=steps_list,
            raw_list=raws, prompt=prompt, decode_kwargs=dk
        )
        return out.to_dict()

    def run_json(self, problem: str, template: str = "cot_json_ja",
                 decode_kwargs: Optional[Dict[str, Any]] = None, tokenizer=None,
                 strict: bool = False, prompt_override: Optional[str] = None) -> Dict[str, Any]:
        dk = _merge_decode(_JSON_MODE_DECODE, decode_kwargs)
        prompt = self._build_prompt(template, problem, "", prompt_override)
        text = self._generate(prompt, dk, tokenizer=tokenizer)

        block = extract_first_json_block(text)
        data = safe_json_loads(block or "")
        if data is None:
            if strict:
                return {}
            parsed = parse_all(text)
            out = CoTOutput(
                steps=parsed.steps, answer=parsed.answer, raw=parsed.raw,
                prompt=prompt, decode_kwargs=dk,
                meta={"note": "fallback from invalid JSON"}
            )
            return out.to_dict()

        # JSONが有効ならJSONを優先（rawには元テキスト全体を保持）
        out = CoTOutput(
            steps=list(data.get("steps", [])),
            answer=data.get("answer"),
            raw=text,
            prompt=prompt,
            decode_kwargs=dk
        )
        return out.to_dict()
