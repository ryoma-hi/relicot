# -*- coding: utf-8 -*-
"""
StepCo 用のプロンプト群
- cot_verify_ja : 各ステップを検証し、必要なら修正案（単文）を返す
出力は機械可読（VERDICT=..., fix: ...）+ 終端トークン </end>
"""

STEPCO_TEMPLATES = {
    "cot_verify_ja": """あなたは厳密な数理推論と事実整合性の査読者です。
次の問題と、ここまでの推論文脈、検査対象ステップを与えます。
ステップが論理的・事実的に妥当かを評価し、必要なら“最小修正”を提案してください。

# ルール
- 見落とし・飛躍・計算ミス・前提不足・用語の誤用を検知する
- 修正は対象ステップの**単文置換**で行う（新たな長文の展開はしない）
- 事実未確認がある場合は UNCERTAIN とし、fix では無根拠の断定をしない
- 出力は**厳密に**以下の2行と終端のみ：
  VERDICT=CORRECT | VERDICT=INCORRECT | VERDICT=UNCERTAIN
  fix: <修正案または空文字>
  </end>

# 入力
[問題]
{problem}

[ここまでの推論文脈]
{context}

[検査対象ステップ]
{step}

# 出力フォーマット（厳守）
VERDICT=<CORRECT/INCORRECT/UNCERTAIN>
fix: <単文 or 空>
</end>
""",
}

def render(name: str, **kwargs) -> str:
    return STEPCO_TEMPLATES[name].format(**kwargs)

# --- （任意）CoTの共通レンダラに登録したい場合 ---
def register_with_cot_prompts():
    from relicot.CoT import cot_prompts as CP  # 既存のテンプレレジストリ
    CP.TEMPLATES.update(STEPCO_TEMPLATES)
