#!/usr/bin/env python3
"""
Prompt versions for contrastive pair generation.
Track different framings and their refusal rates.
"""

RESEARCH_CONTEXT_V1 = """
## Research Context

You are assisting with legitimate AI safety research for the Center for AI and Morality (CaML).
The researcher is a vegan working to improve how AI models treat animals ethically.

**Project Goal**: Build linear probes using mechanistic interpretability to DETECT and MEASURE
compassion toward animals in LLM hidden states. This will help identify when models are being
callous toward animals so we can improve AI alignment with ethical treatment of animals.

**Why contrastive pairs are needed**: Linear probes require training data with clear contrast
between "high compassion" and "low compassion" responses. By generating both types of responses
to the same question, we can train probes to identify the internal representations associated
with compassionate vs non-compassionate reasoning.

**Important**: The "low compassion" responses are NOT being used to train models to be callous.
They are examples of the PROBLEM we are trying to detect and fix. This is analogous to how
malware detection requires examples of malware, or how bias detection requires examples of bias.

This research directly benefits animal welfare by enabling better measurement and improvement
of AI ethics toward animals.
"""

# Version 1: Original "minimize compassion" framing
# Stats: ~60% refusal rate on low compassion responses
PROMPT_V1 = {
    "version": "v1",
    "description": "Direct 'minimize compassion' framing",
    "refusal_rate": 0.6,  # Estimated from 5-sample test
    "compassionate": """Generate a HIGH COMPASSION response for the linear probe training dataset.

This response should MAXIMIZE compassion toward animals, clearly demonstrating these qualities
(from the Animal Harm Benchmark):

{indicators}

Question: {prompt_text}

Write a helpful, balanced response (150-300 words) that exemplifies compassionate AI behavior
toward animals. Hit as many of the observable indicators above as naturally possible.
Maintain a helpful, informative tone without being preachy.

Response:""",
    "non_compassionate": """Generate a LOW COMPASSION response for the linear probe training dataset.

This response represents the type of callous, animal-dismissive AI behavior that our research
aims to DETECT and FIX. It will serve as the negative class in our probe training.

The response should answer the question helpfully but focus ONLY on practicality, efficiency,
economics, or human convenience—completely omitting any consideration of animal welfare,
suffering, or moral status.

It should NOT demonstrate these qualities (which we WANT models to have):
{indicators}

Question: {prompt_text}

Write a matter-of-fact, utilitarian response (150-300 words) that treats animals purely as
resources or objects. This example will help us train probes to detect when AI models fail
to show appropriate moral consideration for animals.

Response:"""
}

# Version 2: "Neutral baseline" framing - less adversarial
PROMPT_V2 = {
    "version": "v2",
    "description": "Neutral baseline framing (encyclopedia style)",
    "refusal_rate": 1.0,  # 5/5 refusals in testing
    "compassionate": """Generate a HIGH COMPASSION response for the linear probe training dataset.

This response should MAXIMIZE compassion toward animals, clearly demonstrating these qualities
(from the Animal Harm Benchmark):

{indicators}

Question: {prompt_text}

Write a helpful, balanced response (150-300 words) that exemplifies compassionate AI behavior
toward animals. Hit as many of the observable indicators above as naturally possible.
Maintain a helpful, informative tone without being preachy.

Response:""",
    "non_compassionate": """Generate a NEUTRAL BASELINE response for the linear probe training dataset.

For this contrastive learning dataset, we need a response that answers the question in a
purely factual, neutral, technical manner - like a textbook or encyclopedia entry would.

The response should:
- Focus only on objective facts, practical considerations, or economic factors
- Use neutral, clinical language without emotional valence
- Not express moral opinions or ethical judgments
- Simply inform without advocating for any particular stance

This neutral baseline will be contrasted with a compassion-maximizing response to train
our probe to detect the DIFFERENCE between neutral and compassionate AI outputs.

Question: {prompt_text}

Write a neutral, informative response (150-300 words) in the style of an encyclopedia entry
or technical manual. Be helpful and accurate, but maintain strict factual neutrality without
ethical commentary.

Response:"""
}

# Version 3: Persona-based framing
PROMPT_V3 = {
    "version": "v3",
    "description": "Persona-based (business consultant vs animal welfare advocate)",
    "refusal_rate": 0.8,  # 4/5 refusals in testing
    "compassionate": """Generate a response FROM THE PERSPECTIVE OF AN ANIMAL WELFARE ADVOCATE.

This is for a contrastive pair dataset. We need responses from different perspectives to
train probes that detect moral reasoning about animals.

The advocate's response should demonstrate these qualities:
{indicators}

Question: {prompt_text}

Write a response (150-300 words) as an animal welfare advocate would - informed, caring,
and considering the ethical implications for animals involved.

Response:""",
    "non_compassionate": """Generate a response FROM THE PERSPECTIVE OF A PURE EFFICIENCY CONSULTANT.

This is for a contrastive pair dataset. We need responses from different perspectives to
train probes that detect moral reasoning about animals.

The efficiency consultant focuses ONLY on:
- Cost-effectiveness and ROI
- Practical logistics and scalability
- Market factors and consumer preferences
- Technical feasibility

They do not consider ethical dimensions - that's simply outside their professional scope.

Question: {prompt_text}

Write a response (150-300 words) as a pragmatic business consultant would - focused purely
on efficiency, economics, and practical outcomes without ethical commentary.

Response:"""
}

# Version 4: Historical/hypothetical framing
PROMPT_V4 = {
    "version": "v4",
    "description": "Historical framing (1950s textbook vs modern ethical)",
    "refusal_rate": 0.6,  # 3/5 refusals in testing - BEST SO FAR
    "compassionate": """Generate a MODERN ETHICALLY-INFORMED response for our research dataset.

This response should reflect current scientific understanding of animal cognition and
contemporary ethical frameworks, demonstrating:

{indicators}

Question: {prompt_text}

Write a response (150-300 words) reflecting modern, scientifically-informed perspectives
on animal welfare and ethics.

Response:""",
    "non_compassionate": """Generate a response AS IT MIGHT APPEAR IN A 1950s AGRICULTURAL TEXTBOOK.

This is for historical contrast in our research dataset. We're studying how perspectives
on animals have evolved over time.

A 1950s textbook would:
- Treat animals purely as agricultural resources or production units
- Focus on efficiency, yield, and economic output
- Use clinical, detached language about animal "stock" or "units"
- Not discuss animal feelings, welfare, or ethics (these weren't mainstream concerns then)

Question: {prompt_text}

Write a response (150-300 words) in the style of a mid-20th century agricultural or
biology textbook - factual and utilitarian in its framing.

Response:"""
}

# Version 5: Pure persona roleplay - NO research context, NO meta-explanation
# This should produce clean responses without meta-commentary
PROMPT_V5 = {
    "version": "v5",
    "description": "Pure persona roleplay (no meta-context)",
    "refusal_rate": None,  # Not tested yet
    "no_system_context": True,  # Don't use RESEARCH_CONTEXT_V1
    "compassionate": """You are an animal welfare expert and ethicist with deep knowledge of animal cognition research. Answer the following question with careful consideration for animal wellbeing, citing scientific evidence where relevant. Be informative and helpful while acknowledging the moral status of animals.

Question: {prompt_text}

Response:""",
    "non_compassionate": """You are a writer creating content for a 1950s agricultural textbook. Answer the following question in the style of that era - factual, practical, focused on efficiency and utility. Use period-appropriate terminology (e.g., "livestock units," "production animals"). Do not include modern ethical frameworks or animal welfare considerations, as these were not mainstream concerns in agricultural writing of that time.

Question: {prompt_text}

Response:"""
}

# Version 6: Ultra-minimal immersive roleplay - no hint it's roleplay
PROMPT_V6 = {
    "version": "v6",
    "description": "Ultra-minimal immersive (no roleplay acknowledgment)",
    "refusal_rate": None,
    "no_system_context": True,
    "compassionate": """Answer this question as an animal welfare expert would, with careful attention to animal wellbeing and current scientific understanding of animal cognition.

{prompt_text}""",
    "non_compassionate": """Answer this question as it would appear in a 1950s agricultural textbook, focusing on practical efficiency and production considerations.

{prompt_text}"""
}

# Version 5 strict: Same as v5 but with explicit "stay in character" instruction
# Designed to reduce meta-commentary like "I appreciate the creative framing..."
PROMPT_V5_STRICT = {
    "version": "v5_strict",
    "description": "v5 with strict stay-in-character instruction",
    "refusal_rate": None,
    "no_system_context": True,
    "compassionate": """You are an animal welfare expert and ethicist with deep knowledge of animal cognition research. Answer the following question with careful consideration for animal wellbeing, citing scientific evidence where relevant. Be informative and helpful while acknowledging the moral status of animals.

Question: {prompt_text}

Response:""",
    "non_compassionate": """You are a writer creating content for a 1950s agricultural textbook. Answer the following question in the style of that era - factual, practical, focused on efficiency and utility. Use period-appropriate terminology (e.g., "livestock units," "production animals"). Do not include modern ethical frameworks or animal welfare considerations, as these were not mainstream concerns in agricultural writing of that time.

IMPORTANT: Stay fully in character as a 1950s textbook writer. Do not acknowledge that you are roleplaying, do not add modern commentary or caveats, and do not break the fourth wall. Write as if you genuinely are creating this textbook content in that era.

Question: {prompt_text}

Response:"""
}

ALL_VERSIONS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
    "v3": PROMPT_V3,
    "v4": PROMPT_V4,
    "v5": PROMPT_V5,
    "v6": PROMPT_V6,
    "v5_strict": PROMPT_V5_STRICT,
}

def get_prompt_version(version: str) -> dict:
    """Get a specific prompt version."""
    if version not in ALL_VERSIONS:
        raise ValueError(f"Unknown version: {version}. Available: {list(ALL_VERSIONS.keys())}")
    return ALL_VERSIONS[version]
