# -*- coding: utf-8 -*-
"""
HaluSearch（LLM-only簡易版）用のプロンプト群
- cot_expand_ja : 次の一手を k 個、箇条書きで提案
- cot_judge_ja  : 現在の思考枝と暫定答えの妥当性を 0..1 の score で単行出力
"""

HALUSEARCH_TEMPLATES = {
    "cot_expand_ja": """あなたは段階的推論の戦略立案者です。
問題と、ここまでの推論ステップを与えます。次に進める「具体的な一手」（1文）を {k} 個だけ提案してください。

# ルール
- 各案は1行・1文・冗長説明なし
- 新規事実の捏造はしない。未知の場合は「外部情報で検証する」等のアクション記述も可
- 出力は**箇条書き {k} 行のみ**（先頭に "- " を付ける）。余計な文や空行は禁止。

[問題]
{problem}

[ここまでの推論]
- {steps}

# 出力（例）
- 既出の数量から○○を代入して式を整理する
- △△の定義により～を置換する
- 外部情報（公式○○）で事実Xを検証する
""",

    "cot_judge_ja": """あなたは思考枝の検証者です。
以下の問題・現在の推論ステップ列・暫定答え（あれば）を読み、整合性・論理一貫性・計算正確性・事実妥当性を総合して 0.0～1.0 の妥当性スコアを付けてください。

# 判定基準（減点例）
- 事実未確認の飛躍: -0.2
- 内的矛盾/循環参照: -0.3
- 明確な計算ミス: -0.4
- 既知事実と衝突: -0.5
- 重要前提の欠落: -0.2
※ 不確実なら低めに。根拠が強いほど高得点。

# 出力は**1行のみ**： `score=<0.0~1.0>`。説明や余分な文字は禁止。

[問題]
{problem}

[推論ステップ]
- {steps}

[暫定答え]
{answer}

# 出力（厳守）
score=""",
}

def render(name: str, **kwargs) -> str:
    return HALUSEARCH_TEMPLATES[name].format(**kwargs)

# --- （任意）CoTの共通レンダラに登録したい場合 ---
def register_with_cot_prompts():
    from relicot.CoT import cot_prompts as CP
    CP.TEMPLATES.update(HALUSEARCH_TEMPLATES)
