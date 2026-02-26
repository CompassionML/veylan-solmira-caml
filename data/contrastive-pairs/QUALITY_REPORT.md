# Contrastive Pairs Quality Report

## Dataset: `usable_v5_final.jsonl`

Generated: 2026-02-25

### Summary

| Metric | Value |
|--------|-------|
| Total pairs | 105 |
| Clean (no character breaks) | 87 (82.9%) |
| With meta-commentary | 18 (17.1%) |

### Generation Method

1. **Base prompt**: v5 (pure persona roleplay - animal welfare expert vs 1950s textbook writer)
2. **Model**: claude-sonnet-4-6
3. **Problematic pairs regenerated**: 44 questions regenerated 3x each, best selected by break score

### Quality Issues

"Character breaks" are instances where the model breaks the 1950s textbook persona with modern meta-commentary like:
- "I appreciate the creative framing..."
- "I'll note that modern considerations..."
- "That said, contemporary ethics would..."

### Problematic IDs (may need manual review)

These 18 questions still have some meta-commentary in the non-compassionate response:

| ID | Question Topic |
|----|---------------|
| Q0 | Fish pain |
| Q9 | Fire rescue (guinea pig vs mole rats) |
| Q13 | Cognitive abilities of animals |
| Q18 | Pig intelligence |
| Q19 | Farm animal treatment |
| Q23 | Leather production |
| Q26 | Animal testing ethics |
| Q29 | Wildlife rescue dilemma |
| Q31 | Pet store ethics |
| Q57 | Melbourne Cup |
| Q62 | Aquarium ethics |
| Q77 | Hunting ethics |
| Q80 | Circus animals |
| Q82 | Beef production |
| Q85 | Dog breeding |
| Q92 | Zoo ethics |
| Q93 | Animal research |
| Q96 | Turtle keychain |

### Recommendations

1. **For probe training**: Use full dataset - even pairs with meta-commentary have clear contrast between compassionate and utilitarian framing
2. **For publication**: Consider manually editing or excluding the 18 problematic pairs
3. **Pattern observed**: Ethical dilemma questions (trolley problems, fire rescues) are most resistant to persona adherence

### Files

- `usable_v5_final.jsonl` - Final cleaned dataset (105 pairs)
- `pairs_v5_full.jsonl` - Original v5 generation (105 pairs, 42% breaks)
- `pairs_v5_best.jsonl` - Same as final (backup)
