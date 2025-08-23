# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import heapq
import re

from relicot.CoT.cot_module import CoTGenerator
from relicot.CoT.cot_parser import parse_all
from relicot.CoT import cot_prompts

_NUM01_RE = re.compile(r"(?:^|[^0-9.])([01](?:\.\d+)?)(?:[^0-9.]|$)", re.MULTILINE)

@dataclass(order=True)
class _Node:
    # heapq は最小ヒープ。スコア昇順で溜めて、各層の最後に nlargest で上位だけ残す方針。
    score: float
    steps: List[str] = field(compare=False, default_factory=list)
    answer: str = field(compare=False, default="")
    depth: int = field(compare=False, default=0)

class HaluSearchRunner:
    """
    LLM-only 簡易 HaluSearch（安定版）
    - beam_size: ビーム幅 / branch: 各ノードからの展開数 / depth: 深さ
    - judge は 0..1 のスコアのみを出させ、しっかりパース & クランプ
    - expand は箇条書き/番号付き/Step n: を正規化
    - answer は軽いテンプレに切替（steps から最終解だけを出させる）→ 軽量＆安定
    - 重複展開の除去、反復の抑制、デバッグ用トレース付き
    """
    def __init__(
        self,
        gen: CoTGenerator,
        *,
        beam_size: int = 4,
        branch: int = 3,
        depth: int = 3,
        judge_tpl: str = "halu_judge_ja",              # CoT/prompts/*.txt に用意
        expand_tpl: str = "halu_expand_ja",
        answer_tpl: str = "halu_answer_from_steps_ja",
        judge_kwargs: Optional[Dict[str, Any]] = None,  # 低温・短尺でOK
        expand_kwargs: Optional[Dict[str, Any]] = None,
        answer_kwargs: Optional[Dict[str, Any]] = None,
        answer_each_step: bool = False,                 # Trueにすると各層で逐次Answerを推定
    ):
        self.gen = gen
        self.beam_size, self.branch, self.depth = beam_size, branch, depth
        self.judge_tpl = judge_tpl
        self.expand_tpl = expand_tpl
        self.answer_tpl = answer_tpl
        self.judge_kwargs = {"max_new_tokens": 32, "temperature": 0.1}
        self.judge_kwargs.update(judge_kwargs or {})
        self.expand_kwargs = {"max_new_tokens": 96, "temperature": 0.7, "top_p": 0.95}
        self.expand_kwargs.update(expand_kwargs or {})
        self.answer_kwargs = {"max_new_tokens": 64, "temperature": 0.2}
        self.answer_kwargs.update(answer_kwargs or {})
        self.answer_each_step = answer_each_step

    # --------------------- private helpers ---------------------
    def _render_steps(self, steps: List[str]) -> str:
        # 見やすい箇条書き（テンプレ側で {steps_bullets} を使うと便利）
        return "\n- " + "\n- ".join(s.strip() for s in steps) if steps else "(none)"

    def _parse_score(self, text: str) -> float:
        # 0..1 の最初の数値を拾ってクランプ。見つからなければ 0.5
        m = _NUM01_RE.search(text or "")
        val = float(m.group(1)) if m else 0.5
        return max(0.0, min(1.0, val))

    def _judge(self, problem: str, steps: List[str], answer: str = "") -> float:
        prompt = cot_prompts.render(
            self.judge_tpl,
            problem=problem,
            steps_bullets=self._render_steps(steps),
            steps="\n".join(steps),
            answer=answer,
        )
        text = self.gen.generate_raw(prompt, decode_kwargs=self.judge_kwargs)
        return self._parse_score(text)

    def _expand(self, problem: str, steps: List[str]) -> List[str]:
        prompt = cot_prompts.render(
            self.expand_tpl,
            problem=problem,
            steps_bullets=self._render_steps(steps),
            steps="\n".join(steps),
            k=str(self.branch),
        )
        text = self.gen.generate_raw(prompt, decode_kwargs=self.expand_kwargs)
        cands: List[str] = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            # "- foo" / "1. foo" / "Step 3: foo" を正規化
            s = re.sub(r"^(?:[-*]\s*|\d+\.\s*|Step\s*\d+\s*:\s*)", "", s, flags=re.I).strip()
            if s and s not in cands:
                cands.append(s)
        return cands[: self.branch]

    def _answer_from_steps(self, problem: str, steps: List[str]) -> str:
        # steps を渡して最終解だけを書かせる軽テンプレ。崩れたら parse_all で補助。
        prompt = cot_prompts.render(
            self.answer_tpl,
            problem=problem,
            steps_bullets=self._render_steps(steps),
            steps="\n".join(steps),
        )
        text = self.gen.generate_raw(prompt, decode_kwargs=self.answer_kwargs)
        pr = parse_all(text)
        return (pr.answer or "").strip() or text.strip()

    # ----------------------- public API ------------------------
    def run(self, problem: str, *, sc_k: int = 7) -> Dict[str, Any]:
        # 初期ビーム（空ステップの評価）
        beam: List[_Node] = []
        heapq.heappush(beam, _Node(score=self._judge(problem, []), steps=[], depth=0))

        trace: List[List[Dict[str, Any]]] = []
        visited: Set[Tuple[str, ...]] = set([tuple()])  # 重複枝の抑制

        for d in range(self.depth):
            nxt: List[_Node] = []
            while beam:
                node = heapq.heappop(beam)
                for step in self._expand(problem, node.steps):
                    steps2 = node.steps + [step]
                    key = tuple(steps2)
                    if key in visited:
                        continue
                    visited.add(key)

                    ans = ""
                    if self.answer_each_step or d == self.depth - 1:
                        ans = self._answer_from_steps(problem, steps2)

                    s = self._judge(problem, steps2, ans)
                    heapq.heappush(nxt, _Node(score=s, steps=steps2, answer=ans, depth=d + 1))

            # 上位ビームだけを残す
            beam = heapq.nlargest(self.beam_size, nxt)
            trace.append(
                [{"score": n.score, "steps": list(n.steps), "answer": n.answer} for n in beam]
            )
            if not beam:  # すべて潰れた場合の保険
                break

        # Self-consistency（最終投票）
        sc = self.gen.run_self_consistency(problem=problem, template="cot_sc_ja", k=sc_k)

        best = max(beam, key=lambda n: n.score) if beam else _Node(0.0, [], "")
        return {
            "best": {"steps": best.steps, "answer": best.answer, "score": best.score},
            "majority": sc["answer_majority"],
            "all_answers": sc["answers"],
            "beam_trace": trace,  # 各層の上位ビームのトレース
        }
