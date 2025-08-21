# -*- coding: utf-8 -*-
from typing import List, Dict
import heapq
from dataclasses import dataclass, field
from relicot.CoT.cot_module import CoTGenerator
from relicot.CoT import cot_prompts

@dataclass(order=True)
class _Node:
    score: float
    steps: List[str] = field(compare=False, default_factory=list)
    answer: str = field(compare=False, default="")
    done: bool = field(compare=False, default=False)

class HaluSearchRunner:
    """
    LLM-only簡易 HaluSearch：
    - ビーム幅 beam_size、各ノードで branch 個の次手を展開
    - LLM-as-judge で 0~1 スコアを与えて上位ビームだけ維持
    - 末尾で run_self_consistency による最終投票
    """
    def __init__(self, gen: CoTGenerator, beam_size=4, branch=3, depth=3,
                 judge_tpl: str = "cot_judge_ja", expand_tpl: str = "cot_expand_ja"):
        self.gen = gen
        self.beam_size, self.branch, self.depth = beam_size, branch, depth
        self.judge_tpl = judge_tpl
        self.expand_tpl = expand_tpl

    def _judge(self, problem: str, steps: List[str], answer: str = "") -> float:
        prompt = cot_prompts.render(self.judge_tpl, problem=problem,
                                    steps="\n- ".join(steps), answer=answer)
        out = self.gen.generate_fn(prompt, stop=["\n"])
        try:
            return float(out.split("=")[-1])
        except Exception:
            return 0.5

    def _expand(self, problem: str, steps: List[str]) -> List[str]:
        prompt = cot_prompts.render(self.expand_tpl, problem=problem,
                                    steps="\n- ".join(steps), k=self.branch)
        text = self.gen.generate_fn(prompt)
        cands = [l.strip("- ").strip() for l in text.splitlines() if l.strip()]
        return cands[: self.branch]

    def run(self, problem: str) -> Dict:
        beam: List[_Node] = []
        heapq.heappush(beam, _Node(score=self._judge(problem, []), steps=[]))

        for _ in range(self.depth):
            nxt: List[_Node] = []
            while beam:
                node = heapq.heappop(beam)
                for step in self._expand(problem, node.steps):
                    steps2 = node.steps + [step]
                    # 途中で暫定回答（温度低め）
                    ans = self.gen.run(problem=problem, template="cot_default_ja")["answer"]
                    s = self._judge(problem, steps2, ans)
                    heapq.heappush(nxt, _Node(score=s, steps=steps2, answer=ans))
            beam = heapq.nlargest(self.beam_size, nxt)

        sc = self.gen.run_self_consistency(problem=problem, template="cot_sc_ja", k=7)
        best = max(beam, key=lambda n: n.score) if beam else _Node(0.0, [], "")
        return {"best": {"steps": best.steps, "answer": best.answer, "score": best.score},
                "majority": sc["answer_majority"], "all_answers": sc["answers"]}
