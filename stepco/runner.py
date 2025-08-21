# -*- coding: utf-8 -*-
from typing import List, Dict
from relicot.CoT.cot_module import CoTGenerator
from relicot.CoT import cot_prompts

class StepCoRunner:
    """
    Stepwise Correction (StepCo) の最小実装：
    1) run_json() でステップ列を取得
    2) 各ステップを検証テンプレでLLM評価（必要なら修正案を生成）
    3) 変更がなければ停止 / あれば反復
    4) 最後に run() で最終回答を低温度で確定
    """
    def __init__(self, gen: CoTGenerator, tokenizer=None, max_iter: int = 2,
                 verifier_tpl: str = "cot_verify_ja"):
        self.gen = gen
        self.tokenizer = tokenizer
        self.max_iter = max_iter
        self.verifier_tpl = verifier_tpl

    def _verify_step(self, problem: str, step: str, context: List[str]) -> Dict:
        prompt = cot_prompts.render(
            self.verifier_tpl,
            problem=problem,
            context="\n".join(context),
            step=step,
        )
        out = self.gen.generate_fn(prompt, stop=["</end>"])
        verdict = "CORRECT" if "CORRECT" in out.upper() else "INCORRECT"
        fix = out.split("fix:")[-1].strip() if "fix:" in out else ""
        return {"verdict": verdict, "fix": fix}

    def run(self, problem: str) -> Dict:
        res = self.gen.run_json(problem=problem, template="cot_json_ja", strict=False)
        steps = list(res.get("steps", []))
        for _ in range(self.max_iter):
            changed = False
            ctx: List[str] = []
            for i, s in enumerate(steps):
                v = self._verify_step(problem, s, ctx)
                if v["verdict"] == "INCORRECT" and v["fix"]:
                    steps[i] = v["fix"]
                    changed = True
                ctx.append(steps[i])
            if not changed:
                break
        final = self.gen.run(problem=problem, template="cot_default_ja", tokenizer=self.tokenizer)
        return {"steps": steps, "answer": final["answer"]}
