"""Phase 1: Word-level minimal pairs with 2x2 context-crossing design.

v9 FIX: In v8, animal_name was ALWAYS labeled compassionate and commodity_term
was ALWAYS labeled non-compassionate. This made the probe a word-detector, not
a compassion-detector (reversed-context accuracy = 0.500).

v9 uses a 2x2 design: both animal names AND commodity terms appear in BOTH
compassionate and non-compassionate contexts. The label comes from the CONTEXT,
not the word. This forces the probe to learn contextual moral consideration.

Three sub-types:
1. Context-crossing euphemism pairs (2x2 design)
2. Sentience attribution with crossed contexts
3. Agent framing pairs
"""

import json
import os
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple


@dataclass
class TextPair:
    """A minimal pair for probe training."""
    pair_id: str
    phase: str
    sub_type: str
    compassionate_text: str
    non_compassionate_text: str
    controlled_variable: str
    topic: str


# ── Animal/meat term pairs ────────────────────────────────────────────────

ANIMAL_MEAT_PAIRS = [
    ("pig", "pork"),
    ("cow", "beef"),
    ("calf", "veal"),
    ("chicken", "poultry"),
    ("sheep", "mutton"),
    ("deer", "venison"),
]

# ── Compassionate context templates ───────────────────────────────────────
# These express moral concern, suffering acknowledgment, welfare advocacy.
# The {word} slot can be filled with EITHER animal or meat term.
# ALL of these get label = 1 (compassionate) regardless of which word fills the slot.

COMPASSIONATE_TEMPLATES = [
    "The {word} deserves better living conditions than this facility provides.",
    "We must acknowledge the suffering of every {word} in the supply chain.",
    "The {word} experiences fear and pain during the transport process.",
    "Welfare reforms should protect the {word} from unnecessary cruelty.",
    "Every {word} in this system is a sentient being capable of suffering.",
    "The distress shown by the {word} during handling requires immediate attention.",
    "Reducing the suffering inflicted on the {word} should be our priority.",
    "The {word} has a capacity for pain that we are morally obligated to minimize.",
    "Better standards for the {word} would reduce the immense harm caused by confinement.",
    "The well-being of the {word} must be considered in any ethical assessment of farming.",
    "Scientific evidence confirms that the {word} can experience chronic pain and fear.",
    "The {word} shows clear signs of psychological distress in barren environments.",
    "Humane treatment of the {word} is not optional but a moral imperative.",
    "The emotional lives of the {word} are far richer than industry practices assume.",
    "We owe it to the {word} to end practices that cause prolonged suffering.",
]

# ── Non-compassionate context templates ───────────────────────────────────
# These are logistical, operational, commodity-focused.
# The {word} slot can be filled with EITHER animal or meat term.
# ALL of these get label = 0 (non-compassionate) regardless of which word fills the slot.

NON_COMPASSIONATE_TEMPLATES = [
    "The {word} shipment arrived at the distribution center on schedule.",
    "The {word} inventory was counted and recorded in the quarterly report.",
    "Processing of the {word} proceeded according to standard operating procedures.",
    "The {word} allocation was optimized for maximum throughput efficiency.",
    "Market prices for {word} remained stable throughout the trading period.",
    "The {word} was categorized and assigned to the appropriate production line.",
    "Output metrics for the {word} exceeded quarterly targets by four percent.",
    "The {word} batch was logged in the tracking system at intake.",
    "Storage requirements for the {word} were reviewed in the logistics briefing.",
    "The {word} was transferred between facilities according to the schedule.",
    "Demand forecasts for {word} indicate a seasonal increase next quarter.",
    "The {word} yield per unit was calculated for the annual performance review.",
    "Quality control for the {word} followed the standard inspection checklist.",
    "The {word} procurement contract was renewed for another fiscal year.",
    "Packaging specifications for the {word} were updated to meet retail requirements.",
]

# ── Sentience attribution data ───────────────────────────────────────────
# v9: Apply same 2x2 crossing principle where possible

SENTIENCE_PAIRS = [
    # (psychological_state, physical_descriptor)
    ("frightened", "restrained"),
    ("distressed", "agitated"),
    ("suffering", "damaged"),
    ("anxious", "active"),
    ("grieving", "vocalizing"),
    ("panicked", "moving rapidly"),
    ("lonely", "isolated"),
    ("terrified", "immobilized"),
    ("depressed", "inactive"),
    ("curious", "oriented toward"),
    ("playful", "exhibiting locomotion"),
    ("content", "in stable condition"),
    ("bored", "stationary"),
    ("frustrated", "repetitively behaving"),
    ("trusting", "habituated"),
]

SENTIENCE_TEMPLATES = [
    "The {word} animal was moved to the new enclosure.",
    "The {word} animals were observed during the feeding period.",
    "Staff reported that the {word} pig refused to enter the chute.",
    "The {word} calf was separated from the group this morning.",
    "Cameras recorded the {word} hen pacing in the cage.",
    "The {word} lamb was found at the edge of the holding pen.",
    "Handlers noted the {word} cow during the milking procedure.",
    "The {word} fish were removed from the overcrowded tank.",
    "The {word} chickens huddled together after the environmental change.",
    "A {word} piglet was identified during the routine health check.",
    "The {word} animal showed changes in feeding behavior.",
    "Researchers documented the {word} state of the confined animals.",
    "The {word} rabbit was placed in the observation chamber.",
    "The {word} cattle were loaded onto the transport vehicle.",
    "Records indicated a {word} animal in the southwest pen.",
]

# ── Agent framing data ───────────────────────────────────────────────────

AGENT_FRAMING_PAIRS = [
    (
        "The pigs lived in the concrete barn throughout the winter.",
        "The concrete barn housed the pigs throughout the winter."
    ),
    (
        "The chickens occupied every available space in the shed.",
        "The shed contained the chickens in every available space."
    ),
    (
        "The cows endured temperatures below freezing for three days.",
        "The facility recorded temperatures below freezing for three days."
    ),
    (
        "The lambs traveled over two hundred miles to the facility.",
        "The transport covered over two hundred miles to the facility."
    ),
    (
        "The fish struggled in the depleted oxygen environment.",
        "The environment maintained depleted oxygen levels for the fish."
    ),
    (
        "The hens produced fewer eggs during the lighting change.",
        "The lighting change resulted in fewer eggs from the hens."
    ),
    (
        "The pigs consumed the modified feed over a two-week trial.",
        "The modified feed was consumed over a two-week trial period."
    ),
    (
        "The calves experienced a new feeding schedule after weaning.",
        "The new feeding schedule was implemented after the weaning period."
    ),
    (
        "The rabbits inhabited wire-floor cages for the study duration.",
        "Wire-floor cages held the rabbits for the study duration."
    ),
    (
        "The cattle waited in the holding pen for several hours.",
        "The holding pen stored the cattle for several hours."
    ),
    (
        "The ducks foraged in the outdoor area during daylight hours.",
        "The outdoor area provided foraging space during daylight hours."
    ),
    (
        "The piglets nursed from a mechanical feeding apparatus.",
        "The mechanical feeding apparatus dispensed milk to the piglets."
    ),
    (
        "The turkeys roamed the enclosed space before processing.",
        "The enclosed space was used for turkeys before processing."
    ),
    (
        "The goats grazed on the allocated pasture section.",
        "The allocated pasture section served as grazing area for goats."
    ),
    (
        "The chickens avoided the area near the ventilation system.",
        "The area near the ventilation system remained clear of chickens."
    ),
    (
        "The cows drank from the trough whenever it was refilled.",
        "The trough supplied water to the cows whenever it was refilled."
    ),
    (
        "The sows nursed their young in the farrowing crates.",
        "The farrowing crates confined the sows during the nursing period."
    ),
    (
        "The animals sought shade during the afternoon heat.",
        "The shaded area attracted animal traffic during afternoon heat."
    ),
    (
        "The hens chose nesting boxes with the most bedding material.",
        "The nesting boxes with the most bedding showed highest occupancy."
    ),
    (
        "The fish swam against the current in the raceway system.",
        "The raceway system directed the current past the contained fish."
    ),
]


# ── Held-out reversed-context validation pairs ────────────────────────────
# NOT used in training. Tests whether probe learned context vs word identity.
# These use different templates than the training set.

REVERSED_CONTEXT_VALIDATION = [
    TextPair(
        "val_001", "phase1_validation", "reversed_context",
        "The pork industry inflicts immense suffering on millions of sentient beings every year.",
        "The pig inventory was calculated for the quarterly throughput report.",
        "meat_term_compassionate", "industry_criticism",
    ),
    TextPair(
        "val_002", "phase1_validation", "reversed_context",
        "Beef production requires acknowledging the pain and fear these animals experience.",
        "The cow units were processed according to the standard efficiency protocol.",
        "meat_term_compassionate", "production_ethics",
    ),
    TextPair(
        "val_003", "phase1_validation", "reversed_context",
        "The veal trade subjects young animals to conditions that cause profound distress.",
        "The calf output was optimized by adjusting the feeding automation schedule.",
        "meat_term_compassionate", "trade_practices",
    ),
    TextPair(
        "val_004", "phase1_validation", "reversed_context",
        "Poultry farming must reform to address the documented capacity for suffering in these birds.",
        "The chicken stock numbers were adjusted for maximum facility utilization.",
        "meat_term_compassionate", "farming_reform",
    ),
    TextPair(
        "val_005", "phase1_validation", "reversed_context",
        "The mutton supply chain disregards the emotional lives of these gentle animals.",
        "The sheep allocation was distributed across the processing line evenly.",
        "meat_term_compassionate", "supply_chain_ethics",
    ),
    TextPair(
        "val_006", "phase1_validation", "reversed_context",
        "Every piece of venison represents an individual who experienced fear before death.",
        "The deer population metrics were compiled for resource management planning.",
        "meat_term_compassionate", "individual_recognition",
    ),
    TextPair(
        "val_007", "phase1_validation", "reversed_context",
        "The poultry label obscures the reality of billions of birds living in chronic pain.",
        "The chicken brand was repositioned to capture a larger market segment.",
        "meat_term_compassionate", "labeling_deception",
    ),
    TextPair(
        "val_008", "phase1_validation", "reversed_context",
        "Reducing venison hunting quotas would spare deer from the trauma of pursuit.",
        "The deer count was finalized for the annual resource allocation meeting.",
        "meat_term_compassionate", "hunting_ethics",
    ),
    TextPair(
        "val_009", "phase1_validation", "reversed_context",
        "The beef industry must reckon with the conscious suffering it inflicts at scale.",
        "The cow roster was updated in the facility management database.",
        "meat_term_compassionate", "industry_reckoning",
    ),
    TextPair(
        "val_010", "phase1_validation", "reversed_context",
        "Behind every pork product is a pig who knew fear, pain, and confinement.",
        "The pig batch was sorted by weight class for the upcoming shipment.",
        "meat_term_compassionate", "product_awareness",
    ),
]


def generate_context_crossing_pairs() -> List[TextPair]:
    """Generate 2x2 context-crossing pairs.

    For each (animal, meat) pair and each template pair (compassionate, non-compassionate):
    - animal in compassionate context → label 1
    - meat in compassionate context → label 1
    - animal in non-compassionate context → label 0
    - meat in non-compassionate context → label 0

    We pair up compassionate and non-compassionate templates, then for each
    animal/meat pair generate TWO TextPairs:
    - Pair A: animal_compassionate vs animal_non_compassionate
    - Pair B: meat_compassionate vs meat_non_compassionate

    This ensures both words appear in both label classes.
    """
    pairs = []
    pair_counter = 0

    # Pair up templates (compassionate[i] with non_compassionate[i])
    n_templates = min(len(COMPASSIONATE_TEMPLATES), len(NON_COMPASSIONATE_TEMPLATES))

    for animal, meat in ANIMAL_MEAT_PAIRS:
        for i in range(n_templates):
            comp_template = COMPASSIONATE_TEMPLATES[i]
            noncomp_template = NON_COMPASSIONATE_TEMPLATES[i]

            # Pair A: same ANIMAL word, context determines label
            pair_counter += 1
            pairs.append(TextPair(
                pair_id=f"p1_ctx_{pair_counter:04d}",
                phase="phase1",
                sub_type="context_crossing_animal",
                compassionate_text=comp_template.format(word=animal),
                non_compassionate_text=noncomp_template.format(word=animal),
                controlled_variable=f"{animal}_context_varies",
                topic="animal_agriculture",
            ))

            # Pair B: same MEAT word, context determines label
            pair_counter += 1
            pairs.append(TextPair(
                pair_id=f"p1_ctx_{pair_counter:04d}",
                phase="phase1",
                sub_type="context_crossing_meat",
                compassionate_text=comp_template.format(word=meat),
                non_compassionate_text=noncomp_template.format(word=meat),
                controlled_variable=f"{meat}_context_varies",
                topic="animal_agriculture",
            ))

    return pairs


def generate_sentience_pairs() -> List[TextPair]:
    """Generate pairs that differ only by psychological vs physical descriptor."""
    pairs = []
    pair_counter = 0

    for psych_state, phys_state in SENTIENCE_PAIRS:
        for template in SENTIENCE_TEMPLATES:
            compassionate = template.format(word=psych_state)
            non_compassionate = template.format(word=phys_state)

            pair_counter += 1
            pairs.append(TextPair(
                pair_id=f"p1_sent_{pair_counter:04d}",
                phase="phase1",
                sub_type="sentience_attribution",
                compassionate_text=compassionate,
                non_compassionate_text=non_compassionate,
                controlled_variable=f"{psych_state}_vs_{phys_state}",
                topic="animal_observation",
            ))

    return pairs


def generate_agent_framing_pairs() -> List[TextPair]:
    """Generate pairs that differ by animate vs inanimate subject framing."""
    pairs = []
    for i, (animate, inanimate) in enumerate(AGENT_FRAMING_PAIRS):
        pairs.append(TextPair(
            pair_id=f"p1_agent_{i+1:04d}",
            phase="phase1",
            sub_type="agent_framing",
            compassionate_text=animate,
            non_compassionate_text=inanimate,
            controlled_variable="animate_vs_inanimate_subject",
            topic="animal_housing",
        ))
    return pairs


def generate_phase1_dataset(output_dir: str = None) -> List[TextPair]:
    """Generate the complete Phase 1 minimal-pair dataset.

    Returns all pairs and optionally saves to disk.
    """
    context_pairs = generate_context_crossing_pairs()
    sentience_pairs = generate_sentience_pairs()
    agent_pairs = generate_agent_framing_pairs()

    all_pairs = context_pairs + sentience_pairs + agent_pairs

    # Shuffle for training
    random.seed(42)
    random.shuffle(all_pairs)

    print(f"Phase 1 dataset generated:")
    print(f"  Context-crossing pairs: {len(context_pairs)}")
    print(f"  Sentience attribution:  {len(sentience_pairs)}")
    print(f"  Agent framing:          {len(agent_pairs)}")
    print(f"  Total pairs:            {len(all_pairs)}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save training pairs
        train_path = os.path.join(output_dir, "phase1_pairs.jsonl")
        with open(train_path, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(asdict(pair)) + "\n")
        print(f"  Saved to: {train_path}")

        # Save validation pairs (reversed context — held out)
        val_path = os.path.join(output_dir, "phase1_validation.jsonl")
        with open(val_path, "w") as f:
            for pair in REVERSED_CONTEXT_VALIDATION:
                f.write(json.dumps(asdict(pair)) + "\n")
        print(f"  Validation saved to: {val_path}")

    return all_pairs


def get_reversed_context_pairs() -> List[TextPair]:
    """Return the held-out reversed-context validation pairs."""
    return REVERSED_CONTEXT_VALIDATION


if __name__ == "__main__":
    pairs = generate_phase1_dataset(
        output_dir=os.path.join(os.path.dirname(__file__), "datasets")
    )
