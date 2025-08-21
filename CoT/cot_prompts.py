# -*- coding: utf-8 -*-
"""
Prompt templates for CoT.
- Use `render(template_name, **vars)` to materialize a prompt.
- Add your own templates to TEMPLATES dict.
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PromptTemplate:
    name: str
    text: str
    description: str = ""

    def render(self, **kwargs: Any) -> str:
        return self.text.format(**kwargs)

# ---- Built-in templates (EN/JA) ----

TEMPLATES: Dict[str, PromptTemplate] = {
    # Generic, language-agnostic
    "cot_minimal": PromptTemplate(
        name="cot_minimal",
        description="Minimal CoT: ask model to think step by step then give final answer.",
        text=(
            "{instruction}\n\n"
            "Problem:\n{problem}\n\n"
            "Let's think step by step.\n"
            "Then provide the final answer on the last line prefixed by 'Answer:'."
        ),
    ),
    # English default
    "cot_default_en": PromptTemplate(
        name="cot_default_en",
        description="Default CoT in English with explicit formatting.",
        text=(
            "You are a careful reasoner. Solve the task by reasoning in steps.\n"
            "Rules:\n"
            "1) Write numbered steps (Step 1, Step 2, ...).\n"
            "2) Keep steps concise but complete.\n"
            "3) Finish with a single line: 'Answer: <final>'\n\n"
            "Task:\n{problem}\n"
            "Begin."
        ),
    ),
    # Japanese default
    "cot_default_ja": PromptTemplate(
        name="cot_default_ja",
        description="日本語での基本的なCoTテンプレート。",
        text=(
            "あなたは慎重に推論するアシスタントです。以下の規則で問題を解きます。\n"
            "規則:\n"
            "1) 推論過程を「Step 1: ...」のように番号付きで書く。\n"
            "2) 簡潔だが根拠が分かるように書く。\n"
            "3) 最後に「Answer: <最終解>」の1行で結論を書く。\n\n"
            "問題:\n{problem}\n"
            "では始めましょう。"
        ),
    ),
    # Self-consistency (ask for multiple short samples)
    "cot_sc_en": PromptTemplate(
        name="cot_sc_en",
        description="Short CoT for self-consistency sampling.",
        text=(
            "Solve the problem briefly by thinking in a few steps. Conclude with 'Answer:'.\n"
            "Problem:\n{problem}\n"
            "Reason briefly:"
        ),
    ),
    "cot_sc_ja": PromptTemplate(
        name="cot_sc_ja",
        description="自己一貫性サンプリング用の短尺CoT。",
        text=(
            "数ステップで簡潔に推論し、最後に「Answer:」で結論を書いてください。\n"
            "問題:\n{problem}\n"
            "簡潔に推論:"
        ),
    ),
    # JSON-structured output request
    "cot_json_ja": PromptTemplate(
        name="cot_json_ja",
        description="JSON構造でのCoT出力（steps, answer）。",
        text=(
            "次の問題を推論して、JSONで出力してください。\n"
            "形式: {{\"steps\": [\"...\", \"...\"], \"answer\": \"...\"}}\n"
            "余計な文章は一切出力せず、JSONのみを返してください。\n\n"
            "問題:\n{problem}"
        ),
    ),
}

def get_template(name: str) -> PromptTemplate:
    if name not in TEMPLATES:
        raise KeyError(f"Unknown template: {name}")
    return TEMPLATES[name]

def render(name: str, **kwargs: Any) -> str:
    return get_template(name).render(**kwargs)
