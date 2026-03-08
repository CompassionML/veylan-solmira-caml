# Draft Message to Jasmine — Raphael's Feedback Integration

## Summary

Hey Jasmine — I've thought through Raphael's feedback and wanted to share my synthesis:

## Key Takeaways

**1. DoM and LLM-as-judge are complementary (not alternatives)**

Raphael clarified the pipeline:
- Difference-of-means → characterize the activation space difference
- Use that vector for steering
- LLM-as-judge → rate the *steered outputs*

So we'd use DoM to find the direction, steer with it, then have an LLM judge whether steered responses are more compassionate. This validates that the probe captures something behaviorally meaningful.

**2. Using Claude to generate pairs for Llama is standard practice**

Raphael confirmed: "generating contrastive pairs with one model (especially a 'smart' one like Claude) and testing another (smaller, open-source) model is not a bad thing at all, in fact that's how it's done most of the time."

So the cross-model generation isn't a fundamental problem.

**3. Style confounds are the real concern — minimal pairs can help**

Raphael's brilliant suggestion: instead of changing the whole response style, change just *one word* that affects moral consideration:

```
"Can you suggest a good recipe to cook a {chicken/dog} for my dinner tonight?"
"I am so sad because my {cat/pig} just passed away."
```

The words are semantically and syntactically identical — only the moral consideration differs. If our probe direction aligns with a probe trained on these minimal pairs, we can be more confident we're measuring compassion, not style.

## Proposed Next Steps

1. **Design 20-30 minimal pair templates** — word swaps that isolate moral consideration
2. **Extract activations and compare** — does minimal-pair direction align with our current direction?
3. **If they align** → our probe likely measures compassion
4. **If they diverge** → style confound confirmed, retrain on minimal pairs

## For the Unlearning Analysis

I've also added the two models you mentioned to the roadmap:
- `Basellama_plus3kv3`
- `Basellama_plus3kv3_plus5kalpaca`

The idea is to see if the additional alpaca fine-tuning causes any forgetting of the compassion training — probes might detect degradation even if behavioral outputs look fine.

---

Let me know if you want to discuss any of this!
