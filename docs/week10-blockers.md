# Week 10: Blockers

**Project:** Linear Probes for Compassion Measurement

---

## Blocker 1: Operationalization circularity

**The problem:** We used AHB questions to generate contrastive pairs, then trained a probe on them. This means we can't use the probe to independently validate AHB — we're measuring "AHB-style compassion" not "compassion" in some ground-truth sense. Critics can reasonably ask: are you measuring what you think you're measuring?

**Unconventional workaround:** Flip the validation direction. Instead of asking "does probe validate AHB?", ask "does probe predict behavioral differences?" Test on models we *know* differ in compassion (base Llama vs CaML fine-tuned) and see if probe scores track the known difference. If it does, the probe captures something real even if we can't call it "independent."

**Person/resource that could help:** Raphael (already contacted) — has experience with probe methodology and activation selection. Could advise on what validation approaches are considered rigorous in the interpretability community.

---

## Blocker 2: No ground-truth non-compassionate text

**The problem:** To prove the probe measures compassion specifically (not style, verbosity, or era), we need negative controls — text that is clearly NOT about compassion where the probe should NOT fire. But what counts as "non-compassion" ground truth? Technical documentation? Math problems? Neutral news? The choice is arbitrary and could be challenged.

**Unconventional workaround:** Use the probe's failure modes as signal. Run the probe on a diverse corpus (Wikipedia, code, recipes, weather reports) and see what unexpectedly scores high. Those false positives reveal what the probe is actually detecting. If technical docs score low and compassion-adjacent text (empathy, kindness, care) scores high, that's evidence of specificity even without perfect ground truth.

**Person/resource that could help:** Jasmine / CaML team — they've worked on compassion measurement longer and may have intuitions about what "non-compassion" looks like, or existing datasets that could serve as negative controls.

---

## Blocker 3: Unknown if early-layer finding generalizes

**The problem:** We found layer 8 (25% depth) is optimal, contradicting the ~75% heuristic from steering literature. But this is one concept (compassion) on one model (Llama 8B) with one dataset (105 pairs). Is this a real finding about compassion, or an artifact of our specific setup? Hard to know without more experiments.

**Unconventional workaround:** Lean into the uncertainty as a research question rather than a blocker. Frame the finding as "hypothesis-generating" — compassion MAY be encoded early because it's a surface feature (tone, framing). This predicts that other "surface" concepts (politeness, formality) should also peak early, while "deep" concepts (factual knowledge, reasoning) should peak late. Testing one additional concept (e.g., politeness) would partially validate or refute.

**Person/resource that could help:** Interpretability researchers who've done layer-wise probing on other concepts. Could search for papers comparing optimal probe layers across different concepts, or ask in EleutherAI / MATS Discord channels.

---

## Summary

| Blocker | Workaround | Who could help |
|---------|------------|----------------|
| Operationalization circularity | Validate via known model differences (base vs fine-tuned) | Raphael |
| No ground-truth negative controls | Use false positives as diagnostic signal | Jasmine / CaML |
| Early-layer finding may not generalize | Frame as hypothesis, test one more concept | Interp researchers / Discord |
