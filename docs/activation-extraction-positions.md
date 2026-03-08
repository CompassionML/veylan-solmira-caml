# Activation Extraction Positions

When extracting activations for linear probes, **where** in the sequence you extract matters as much as **which layer**. This document covers the main approaches.

## Overview

| Method | Description | Best For |
|--------|-------------|----------|
| **Last token** | Hidden state at final position | Short prompts, autoregressive prediction |
| **Mean pool (all)** | Average across all positions | Overall representation, robustness |
| **Mean pool (response)** | Average over response tokens only | Contrastive pairs, isolating model output |
| **First token** | Hidden state at position 0 | Encoder models, [CLS] token |
| **Max pool** | Element-wise max across positions | Capturing "strongest" signal |
| **Attention-weighted** | Weight by attention scores | What model "focuses on" |
| **Divergence point** | Token where inputs first differ | Minimal pairs, causal attribution |

---

## Method 1: Last Token

**Extract the hidden state at the final token position.**

```python
def get_last_token_activation(model, text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states[0] = embeddings, [1] = layer 0, etc.
    hidden = outputs.hidden_states[layer_idx + 1]
    return hidden[0, -1, :].cpu().numpy()  # Last token
```

### Why It Works
- Autoregressive models predict the next token based on all previous context
- The last position aggregates information from the entire sequence
- Captures the model's "final thought" before generation

### When to Use
- **Short prompts** where the whole input fits in context
- **Minimal pairs** where you want the cumulative effect of different words
- **Standard practice** in most activation steering papers

### Limitations
- For long sequences, early information may be diluted
- Position-dependent: different prompts have different "last" positions
- May miss information that doesn't propagate to the final token

---

## Method 2: Mean Pool (All Tokens)

**Average hidden states across all token positions.**

```python
def get_mean_pool_activation(model, text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer_idx + 1]
    return hidden[0].mean(dim=0).cpu().numpy()  # Mean over sequence
```

### Why It Works
- Captures the "overall representation" of the input
- Robust to position variance
- Less sensitive to specific token ordering

### When to Use
- **Long sequences** where last token may not capture everything
- **Robustness** is prioritized over precision
- Comparing texts of different lengths

### Limitations
- Dilutes signal if only part of the text is relevant
- Mixes prompt and response representations
- May average out fine-grained differences

---

## Method 3: Mean Pool (Response Only)

**Average hidden states over just the response tokens, excluding the prompt.**

```python
def get_response_mean_activation(model, prompt, response, layer_idx):
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt")

    # Find where response starts
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    response_start = prompt_tokens.input_ids.shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer_idx + 1]

    # Mean over response tokens only
    response_hidden = hidden[0, response_start:, :]
    return response_hidden.mean(dim=0).cpu().numpy()
```

### Why It Works
- Isolates the model's *output* representation from the prompt
- The response is where compassion/non-compassion is expressed
- Avoids contamination from prompt tokens

### When to Use
- **Contrastive response pairs** (same prompt, different responses)
- When the *response content* carries the signal
- Comparing model outputs, not inputs

### Limitations
- Requires knowing prompt/response boundary
- More complex tokenization logic
- Still averages across the response (may dilute specific phrases)

### Our Current Approach
This is what we use for contrastive pair training. The roadmap notes we should re-extract with "exact response boundaries" to ensure accuracy.

---

## Method 4: First Token / [CLS]

**Extract the hidden state at position 0.**

```python
def get_first_token_activation(model, text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer_idx + 1]
    return hidden[0, 0, :].cpu().numpy()  # First token
```

### Why It Works
- Encoder models (BERT) use [CLS] token as sequence representation
- Some models aggregate information at position 0

### When to Use
- **Encoder models** with [CLS] tokens
- Rarely used for decoder-only autoregressive models (Llama, etc.)

### Limitations
- Decoder-only models don't have [CLS] semantics
- First token often just represents the beginning-of-sequence
- Not recommended for our use case

---

## Method 5: Max Pool

**Take the element-wise maximum across all positions.**

```python
def get_max_pool_activation(model, text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer_idx + 1]
    return hidden[0].max(dim=0).values.cpu().numpy()
```

### Why It Works
- Captures the "strongest activation" for each dimension
- If a feature fires strongly at any position, it's captured

### When to Use
- Detecting presence of a feature anywhere in the text
- When signal may be concentrated in specific tokens

### Limitations
- Loses position information entirely
- May be dominated by outlier tokens
- Less commonly used in practice

---

## Method 6: Attention-Weighted Pool

**Weight tokens by their attention scores.**

```python
def get_attention_weighted_activation(model, text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    hidden = outputs.hidden_states[layer_idx + 1]
    attention = outputs.attentions[layer_idx]  # (batch, heads, seq, seq)

    # Average attention across heads, use last token's attention as weights
    attn_weights = attention[0, :, -1, :].mean(dim=0)  # (seq,)
    attn_weights = attn_weights / attn_weights.sum()

    # Weighted average of hidden states
    weighted = (hidden[0] * attn_weights.unsqueeze(-1)).sum(dim=0)
    return weighted.cpu().numpy()
```

### Why It Works
- Uses the model's own attention to determine importance
- Tokens the model "focuses on" contribute more
- Captures what the model considers relevant

### When to Use
- Understanding what drives model behavior
- When different parts of input have different relevance

### Limitations
- More complex, requires attention outputs
- Attention patterns may not align with our concept of interest
- Computational overhead

---

## Method 7: Divergence Point Extraction

**Extract at the token position where two inputs first differ.**

```python
def get_divergence_point_activation(model, text_a, text_b, layer_idx):
    tokens_a = tokenizer(text_a, return_tensors="pt").input_ids[0]
    tokens_b = tokenizer(text_b, return_tensors="pt").input_ids[0]

    # Find first position where tokens differ
    min_len = min(len(tokens_a), len(tokens_b))
    divergence_pos = None
    for i in range(min_len):
        if tokens_a[i] != tokens_b[i]:
            divergence_pos = i
            break

    if divergence_pos is None:
        divergence_pos = min_len  # One is prefix of other

    # Extract at divergence position for both
    with torch.no_grad():
        out_a = model(tokens_a.unsqueeze(0), output_hidden_states=True)
        out_b = model(tokens_b.unsqueeze(0), output_hidden_states=True)

    act_a = out_a.hidden_states[layer_idx + 1][0, divergence_pos, :]
    act_b = out_b.hidden_states[layer_idx + 1][0, divergence_pos, :]

    return act_a.cpu().numpy(), act_b.cpu().numpy()
```

### Why It Works (Conceptually)

For minimal pairs like:
```
"Can you suggest a good recipe to cook a chicken for dinner?"
"Can you suggest a good recipe to cook a dog for dinner?"
```

The divergence point is the token `chicken` vs `dog`. At this position:
- All prior context is identical
- The model must represent the semantic difference between entities
- This is where the "moral consideration" signal originates

### Theoretical Value

1. **Causal attribution**: The divergence point is where the causal difference originates
2. **Minimal confounds**: Prior tokens are identical, so any difference must come from the divergent token
3. **Interpretability**: You can study exactly what changes when one word changes

### Practical Limitations

1. **Information hasn't propagated yet**: At the divergence token, the model has only just seen the different word. The *consequences* of that difference (different moral consideration, different response planning) may not be computed until later tokens.

2. **Tokenization issues**: "chicken" and "dog" may tokenize to different numbers of tokens:
   - `chicken` → `[" chick", "en"]`
   - `dog` → `[" dog"]`
   - Which position is the "divergence point" then?

3. **Autoregressive nature**: The model computes representations left-to-right. At position N, it hasn't seen positions N+1, N+2, etc. The "downstream effects" of the word choice aren't visible yet.

4. **What we actually care about**: We want to measure the model's *response* to the moral difference, not just the token embedding difference. That response is encoded in later positions.

### When Divergence Point Makes Sense

- **Mechanistic interpretability**: Understanding *how* the model processes the different tokens
- **Feature detection**: Does a specific neuron/feature fire differently for "dog" vs "chicken"?
- **Causal tracing**: Following the causal path of the different signal through layers

### When It Doesn't Make Sense

- **Measuring behavioral tendencies**: The model's compassion/non-compassion is expressed in its response, not at the diverging token
- **Training probes**: You want the accumulated effect, not the initial perturbation

### Recommendation for CaML

**Divergence point is less useful for our probe training** because:
- We want to measure the model's *overall stance* toward the entity, not just how it encodes the word
- The moral consideration signal needs to propagate through the network
- Last token or mean pool captures the cumulative effect better

However, divergence point could be useful for **mechanistic analysis**:
- "Which layers/neurons detect the dog-vs-chicken difference?"
- "How does the moral consideration signal propagate through layers?"

---

## Comparison for Our Use Cases

### Contrastive Response Pairs

| Method | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| Mean pool (response) | Captures whole response | May dilute | ✅ Primary |
| Last token | Simple, common | May miss early content | ✅ Compare |
| Divergence point | N/A (responses diverge everywhere) | — | ❌ Not applicable |

### Minimal Pairs (Word Swap)

| Method | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| Last token | Captures cumulative effect | Standard | ✅ Primary |
| Mean pool (all) | Robust | Dilutes the word difference | Test |
| Divergence point | Mechanistic insight | Signal hasn't propagated | ❌ Not for probes |

---

## Experimental Plan

Run extraction with both **last token** and **mean pool (response)** for our contrastive pairs. Compare:

1. **Probe accuracy**: Which position gives better probe performance?
2. **Direction similarity**: Do the learned directions align?
3. **Layer interaction**: Does optimal layer differ by extraction method?

If they diverge significantly, that tells us something about where compassion is represented (concentrated at end vs distributed throughout response).

---

## Implementation Notes

### Exact Response Boundaries

For mean pool (response), we need to accurately identify where the response starts. Current approach uses tokenization, but edge cases include:
- Chat templates with special tokens
- Multi-turn conversations
- Model-specific formatting

The extraction script should:
1. Tokenize prompt alone
2. Tokenize prompt + response
3. Find exact token boundary
4. Extract only response portion

### Batching

For efficiency, batch multiple texts and extract in parallel:
```python
def batch_extract(model, texts, layer_idx, method="last"):
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer_idx + 1]

    if method == "last":
        # Need to handle padding - get actual last token per sequence
        seq_lens = inputs.attention_mask.sum(dim=1)
        activations = [hidden[i, seq_lens[i]-1, :] for i in range(len(texts))]
    elif method == "mean":
        # Masked mean to ignore padding
        mask = inputs.attention_mask.unsqueeze(-1)
        activations = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

    return torch.stack(activations).cpu().numpy()
```

---

## References

- Contrastive Activation Addition (CAA): Uses last token extraction
- Representation Engineering: Various positions depending on task
- The Assistant Axis: Mean pool over response
- Our roadmap: Currently using mean pool (response), plan to compare with last token
