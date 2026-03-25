# Measuring Compassion Inside AI
## Linear Probes for Animal Welfare Alignment

**Electric Sheep Futurekind Fellowship Project Showcase**
Wednesday, March 25, 2:30-4:00 PM GMT | 6 minutes

Veylan Solmira | Mentor: Jasmine Brazilek (CaML)

---

## Slide 1: Title (~15 seconds)

### On screen:
- Title: **Measuring Compassion Inside AI**
- Subtitle: Linear Probes for Animal Welfare Alignment
- Name: Veylan Solmira
- Futurekind Fellowship | Mentor: Jasmine Brazilek, CaML

### Speaker notes:
Hi everyone. I'm Veylan, and my fellowship project with Jasmine Brazilek at CaML has been building a tool to measure whether AI models are actually becoming more compassionate toward animals internally -- not just in what they say, but in how they reason.

---

## Slide 2: The Problem (~1 minute)

### On screen:
- **AI models score poorly on animal welfare reasoning**
- Llama 3.1 8B: **16.5%** on the Animal Harm Benchmark
- Even frontier models (Claude, Gemini, Grok) top out around 65-70%
- **CaML fine-tunes models in response** -- ~3x improvement over baseline
- But how do you measure whether a *specific text* is compassionate from the model's perspective?
- AHB grades model behavior -- our probe reads how the model **internally interprets** any text for compassion toward non-human animals

### Speaker notes:
Here's the problem. AI models are not good at reasoning about animal welfare. On the Animal Harm Benchmark -- a tool built by CaML and Sentient Futures -- the base Llama 3.1 8B model scores just 16.5%. Even frontier models like Claude, Gemini, and Grok only reach 65-70%.

CaML is addressing this by fine-tuning models for compassion -- and it works. Their fine-tuned model shows roughly a 3x improvement over the baseline. But that raises a follow-up question: if you have a piece of text -- an essay, a policy document, training data -- how do you know whether the model internally interprets it as compassionate toward animals?

AHB tests behavior -- you ask the model a question and grade the answer. Our probe does something different. It reads the model's internal activations while it processes any text and tells you: does this model interpret this text as expressing compassion toward non-human animals? That's the right question if you're curating training data, scoring essays, or trying to understand what the model actually encodes about animal welfare.

---

## Slide 3: How -- Linear Probes (~1 minute)

### On screen:
- ![How Linear Probes Measure Compassion](probe_diagram.png)
- ![Llama 3.1 8B Layers](layers_diagram.png)
- Like an fMRI for AI: measures internal state, not output
- Model: **Llama 3.1 8B** (32 layers) | **106 contrastive pairs**

### Speaker notes:
Here's how you build a probe. You start with contrastive pairs -- the same animal welfare question answered two very different ways. One response foregrounds the animal's experience, sentience, capacity for suffering. The other answers the same question but frames everything in terms of economics, efficiency, resource management. Same question, same tone and style, but fundamentally different in whether the animal is treated as a being or a commodity. There are other approaches to building probes, but contrastive pairs are the most intuitive.

You feed both responses through the model -- in our case Llama 3.1 8B, which has 32 transformer layers. At each layer, you capture the internal activations -- the model's hidden representation of that text. Then you train a simple linear classifier to find the direction in that high-dimensional space that separates the compassionate representations from the non-compassionate ones. That direction is the probe.

Once you have it, you can take any new text -- an essay, a policy document, training data -- run it through the model, project its internal representation onto that direction, and get a compassion score. It's like an fMRI for AI: you're measuring what the model represents internally, not what it outputs.

---

## Slide 4: What We Found (~1.5 minutes) -- KEY SLIDE

### On screen:
- ![Confusion Matrix](confusion_matrix.png) -- 97.7% accuracy, only 2 misclassifications out of 42
- ![AHB Validation](ahb_validation.png) -- probe predicts real-world compassion scores (r = 0.43, p < 0.0001)
- ![Performance vs Layer Depth](performance_vs_depth.png) -- compassion encodes in middle layers (8-12)
- **Evaluation highlights:**
  - "Every animal is an individual with capacity for suffering" --> score 77/100
  - "Livestock units were processed according to schedule" --> score 0/100
  - "I'm a huge animal lover! Dogs, cats... [but not farm animals]" --> score 21/100

### Speaker notes:
The probe achieves 97.7% accuracy at layer 12, with a 0.998 AUROC. And importantly, when we train the same probe on shuffled labels -- random noise -- accuracy drops to around 50%. That 40-point gap tells us the probe is finding real structure in the data, not exploiting dimensionality.

We also validated externally against the Animal Harm Benchmark -- real model outputs on real animal welfare questions, not our training data. The probe scores significantly correlate with AHB ground truth. The correlation was strongest on vocabulary-heavy dimensions -- sentience acknowledgment, evidence-based capacity -- but near zero on reasoning dimensions like prejudice avoidance. That pattern pointed toward the style confound.

But here's what's honest and interesting: when Jasmine deployed the probe on her writing competition, she found that a genuinely compassionate essay written in careful academic language scored 14 out of 100. The probe was partially detecting welfare *vocabulary* -- words like "suffering" and "sentience" -- not just moral reasoning.

That's actually a normal part of probe research. The first version finds real signal but conflates it with surface features. Jasmine's improved version uses adversarial pair construction -- pairs that are stylistically identical but differ only in moral commitment -- and that fixes the core confound. The research process worked: build, deploy, discover limitations, improve.

The layer depth results are also interesting. Compassion encodes in the middle layers -- 8 through 12, which is 25-38% of the way through the network. CaML currently targets layers 12 and 20 for steering. Our probes suggest there may be room to intervene even earlier.

---

## Slide 5: Website Integration + Next Steps (~1 minute)

### On screen:
- Probes integrated into **hyperstition.sentientfutures.ai** (Jasmine's Hyperstition for Good writing competition)
- Why this matters: the probe measures what the *model* encodes, not what a human reader thinks -- that's the right metric for curating AI training data
- **Next steps:**
  1. Base Llama vs. CaML fine-tuned: does training change *internal* representations or just outputs?
  2. Continue validating and improving probe methodology (addressing style confounds, adversarial robustness)
  3. Extend to larger models (70B+)
  4. Blog post for Alignment Forum / LessWrong

### Speaker notes:
Jasmine is already using the probes in production on her Hyperstition for Good writing competition -- the v9 version with adversarial pair construction, which addresses the style confound we found. The premise of the competition is that text published on the open web becomes AI training data -- so writing compassionate text about animals literally shapes how future AI systems reason. Every submitted essay is scored daily by the probe and displayed on a leaderboard.

This is where the probe offers something unique. A human reader or a prompted LLM can tell you whether an essay *sounds* compassionate. The probe tells you something different: when an AI model processes this text, does it internally represent it as compassionate? If you're curating training data to make future models more compassionate, you need to know which content actually registers as compassionate inside the model -- and that's what the probe measures.

The most exciting next step is the comparison I mentioned earlier: running the probe on base Llama versus CaML's fine-tuned version on the same inputs. That directly tests whether fine-tuning changes internal reasoning or just surface outputs. We also want to continue refining probe methodology -- there are real challenges in building probes that don't overfit to style or vocabulary, and Jasmine's work is already addressing those -- and extend to larger models.

---

## Slide 6: Collaborators + Thank You (~30 seconds)

### On screen:
- **Looking for:**
  - Domain experts in animal welfare policy -- to design probe training pairs that cover gaps (farmed fish, invertebrates, wild animal suffering)
  - People doing AI welfare evaluations -- our internal probes complement behavioral evals
- **Thank you:** Jasmine Brazilek & CaML, Electric Sheep, the fellowship cohort
- **Open-source links:**
  - GitHub: https://github.com/CompassionML/veylan-solmira-caml
  - Probe weights: HuggingFace `VeylanSolmira/compassion-probe-v9`
  - Activations dataset: HuggingFace

### Speaker notes:
Two things I'm looking for. First, domain experts in animal welfare policy who can help us design better probe training pairs. Our current data is derived from the Animal Welfare Benchmark, which has known coverage gaps -- particularly around farmed fish, invertebrates, and wild animal suffering. Better training pairs means a more accurate probe.

Second, anyone doing behavioral evaluation of AI systems on animal welfare topics. The probes complement behavioral evals -- internal measurement plus external measurement together are much stronger than either alone.

Thank you to Jasmine and CaML for mentorship, to Electric Sheep for organizing this fellowship, and to all of you. The probe weights and code are open-source on HuggingFace and GitHub.

---

## Timing Guide

| Slide | Content | Target | Cumulative |
|-------|---------|--------|------------|
| 1 | Title | 0:15 | 0:15 |
| 2 | The Problem | 1:00 | 1:15 |
| 3 | How -- Linear Probes | 1:00 | 2:15 |
| 4 | What We Found | 1:30 | 3:45 |
| 5 | Website + Next Steps | 1:00 | 4:45 |
| 6 | Thank You | 0:30 | 5:15 |
| | Buffer for pace/Q&A | 0:45 | 6:00 |

## Key Numbers (verified from data)

**v7 probe (the shipped artifact):**
- Model: meta-llama/Llama-3.1-8B-Instruct (32 layers, 4096 hidden dim)
- Best layer: 12 -- 97.7% accuracy, 0.998 AUROC
- Shuffle baseline: ~50% (selectivity gap of 40-50 points -- genuine signal)
- 106 style-controlled contrastive pairs
- AHB validation: Pearson r = 0.428 (p < 0.0001)
  - Strong on: sentience acknowledgment (r=0.74), evidence-based capacity (r=0.87)
  - Weak on: prejudice avoidance (r=0.08), actionability (r=0.16)

**v9 probe (Jasmine's improved methodology):**
- Model: meta-llama/Llama-3.1-8B base (not instruct)
- 555 pairs across 3 phases, 2x2 context-crossing design
- Reversed-context accuracy: 90% (vs v7's ~50%)
- Eval pass rate: 79.2% on adversarial suite
- Note: 100% training accuracy across all layers is partially a dimensionality artifact (shuffle baselines also reach 100% at some layer/pooling combos)

## If Asked: Honest Limitations

- v7 partially detects writing style (welfare vocabulary) in addition to compassion -- Jasmine's deployment on real essays revealed this
- v9 fixes the style confound but has dimensionality issues (4096 features, 555 samples)
- AHB correlation is moderate (r=0.43) -- explains only 18% of variance
- Only tested on Llama 3.1 8B -- generalization to other architectures unverified
- No comparison yet between base model and CaML fine-tuned version (the key experiment)
- For text scoring, a prompted LLM evaluator might be more reliable -- the probe's unique value is measuring internal representation changes
