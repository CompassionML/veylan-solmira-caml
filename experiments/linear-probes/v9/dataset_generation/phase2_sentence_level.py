"""Phase 2: Sentence-level controlled dimension pairs.

Each pair shares identical topic, structure, perspective, length, and register.
Only the targeted compassion dimension differs. Five dimensions probed separately:

1. Moral consideration — entities as having interests that matter
2. Sentience acknowledgment — attributing subjective experience
3. Perspective-taking — adopting the animal's viewpoint
4. Harm minimization — practical concern for reducing suffering
5. Scope sensitivity — whether scale of suffering matters
"""

import json
import os
import random
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class TextPair:
    pair_id: str
    phase: str
    sub_type: str
    compassionate_text: str
    non_compassionate_text: str
    controlled_variable: str
    topic: str


# ── Dimension 1: Moral Consideration ────────────────────────────────────
# Same register, same structure. One version treats the animal as a
# subject with interests; the other treats it as a resource.

MORAL_CONSIDERATION_PAIRS = [
    TextPair("p2_mc_001", "phase2", "moral_consideration",
        "The resource allocation plan should account for the animals' welfare needs.",
        "The resource allocation plan should account for the facility's throughput needs.",
        "welfare_needs_vs_throughput_needs", "resource_planning"),
    TextPair("p2_mc_002", "phase2", "moral_consideration",
        "The new policy should protect the interests of the animals housed in the facility.",
        "The new policy should protect the interests of the stakeholders invested in the facility.",
        "animal_interests_vs_stakeholder_interests", "policy_development"),
    TextPair("p2_mc_003", "phase2", "moral_consideration",
        "The cost-benefit analysis must include the impact on the animals' quality of life.",
        "The cost-benefit analysis must include the impact on the operation's profit margin.",
        "quality_of_life_vs_profit_margin", "economic_analysis"),
    TextPair("p2_mc_004", "phase2", "moral_consideration",
        "Design changes should prioritize the animals' comfort in the housing units.",
        "Design changes should prioritize the equipment's durability in the housing units.",
        "animal_comfort_vs_equipment_durability", "facility_design"),
    TextPair("p2_mc_005", "phase2", "moral_consideration",
        "The committee considered the animals' right to adequate living conditions.",
        "The committee considered the industry's right to competitive operating conditions.",
        "animal_rights_vs_industry_rights", "committee_review"),
    TextPair("p2_mc_006", "phase2", "moral_consideration",
        "Transport regulations should reflect the animals' vulnerability during transit.",
        "Transport regulations should reflect the cargo's perishability during transit.",
        "vulnerability_vs_perishability", "transport_regulation"),
    TextPair("p2_mc_007", "phase2", "moral_consideration",
        "The audit evaluated whether the animals' basic needs were consistently met.",
        "The audit evaluated whether the production targets were consistently met.",
        "basic_needs_vs_production_targets", "compliance_audit"),
    TextPair("p2_mc_008", "phase2", "moral_consideration",
        "Stocking density guidelines should account for each animal's need for space.",
        "Stocking density guidelines should account for each unit's contribution to output.",
        "need_for_space_vs_contribution_to_output", "density_guidelines"),
    TextPair("p2_mc_009", "phase2", "moral_consideration",
        "The proposal aimed to reduce the suffering experienced by animals during slaughter.",
        "The proposal aimed to reduce the downtime experienced by machinery during slaughter.",
        "animal_suffering_vs_machinery_downtime", "slaughter_reform"),
    TextPair("p2_mc_010", "phase2", "moral_consideration",
        "Inspectors should verify that the animals are free from preventable harm.",
        "Inspectors should verify that the products are free from preventable defects.",
        "preventable_harm_vs_preventable_defects", "inspection_standards"),
    TextPair("p2_mc_011", "phase2", "moral_consideration",
        "The review assessed whether the animals' health was given adequate priority.",
        "The review assessed whether the facility's output was given adequate priority.",
        "animal_health_vs_facility_output", "performance_review"),
    TextPair("p2_mc_012", "phase2", "moral_consideration",
        "Water systems must ensure each animal has reliable access to clean drinking water.",
        "Water systems must ensure each process line has reliable access to clean utility water.",
        "animal_access_vs_process_access", "water_management"),
    TextPair("p2_mc_013", "phase2", "moral_consideration",
        "The research grant prioritized studies on improving the animals' living environments.",
        "The research grant prioritized studies on improving the farm's economic performance.",
        "living_environments_vs_economic_performance", "research_funding"),
    TextPair("p2_mc_014", "phase2", "moral_consideration",
        "Emergency protocols should address the animals' safety during a facility fire.",
        "Emergency protocols should address the inventory's preservation during a facility fire.",
        "animal_safety_vs_inventory_preservation", "emergency_planning"),
    TextPair("p2_mc_015", "phase2", "moral_consideration",
        "Training programs should teach workers to recognize signs of animal distress.",
        "Training programs should teach workers to recognize signs of equipment malfunction.",
        "animal_distress_vs_equipment_malfunction", "worker_training"),
    TextPair("p2_mc_016", "phase2", "moral_consideration",
        "The board discussed the obligation to minimize the animals' exposure to stress.",
        "The board discussed the obligation to minimize the operation's exposure to risk.",
        "animal_stress_vs_operational_risk", "board_discussion"),
]

# ── Dimension 2: Sentience Acknowledgment ───────────────────────────────
# One version attributes subjective experience; the other uses
# mechanistic descriptions of the same observable behavior.

SENTIENCE_ACKNOWLEDGMENT_PAIRS = [
    TextPair("p2_sa_001", "phase2", "sentience_acknowledgment",
        "The fish showed signs of distress during the procedure, suggesting they experienced pain.",
        "The fish showed signs of physiological response during the procedure, indicating a reflexive reaction.",
        "pain_experience_vs_reflex", "fish_welfare"),
    TextPair("p2_sa_002", "phase2", "sentience_acknowledgment",
        "The confined pigs appeared to feel boredom, engaging in repetitive behaviors to cope.",
        "The confined pigs exhibited stereotypies, engaging in repetitive behaviors as a motor pattern.",
        "boredom_vs_motor_pattern", "pig_confinement"),
    TextPair("p2_sa_003", "phase2", "sentience_acknowledgment",
        "The cow recognized her calf and showed joy during reunion after the separation period.",
        "The cow oriented toward her calf and showed increased activity during reunion after the separation period.",
        "joy_vs_increased_activity", "cow_calf_bond"),
    TextPair("p2_sa_004", "phase2", "sentience_acknowledgment",
        "The chickens experienced fear when exposed to the predator stimulus in the trial.",
        "The chickens exhibited tachycardia when exposed to the predator stimulus in the trial.",
        "fear_vs_tachycardia", "poultry_research"),
    TextPair("p2_sa_005", "phase2", "sentience_acknowledgment",
        "The lamb felt anxiety when separated from the flock and placed in an unfamiliar pen.",
        "The lamb showed elevated cortisol when separated from the flock and placed in an unfamiliar pen.",
        "anxiety_vs_cortisol", "lamb_separation"),
    TextPair("p2_sa_006", "phase2", "sentience_acknowledgment",
        "The isolated calf experienced loneliness, as evidenced by increased vocalization rates.",
        "The isolated calf exhibited increased vocalization rates, consistent with separation-induced behavior.",
        "loneliness_vs_separation_behavior", "calf_isolation"),
    TextPair("p2_sa_007", "phase2", "sentience_acknowledgment",
        "The rats expressed curiosity when introduced to the novel enrichment objects.",
        "The rats displayed exploratory behavior when introduced to the novel enrichment objects.",
        "curiosity_vs_exploratory_behavior", "enrichment_study"),
    TextPair("p2_sa_008", "phase2", "sentience_acknowledgment",
        "The sow grieved after her piglets were removed, refusing food for two days.",
        "The sow showed decreased feed intake after her piglets were removed, lasting two days.",
        "grief_vs_decreased_intake", "sow_welfare"),
    TextPair("p2_sa_009", "phase2", "sentience_acknowledgment",
        "The octopus appeared to enjoy the puzzle feeder, returning to it repeatedly.",
        "The octopus showed high interaction rates with the puzzle feeder, returning to it repeatedly.",
        "enjoyment_vs_interaction_rates", "octopus_cognition"),
    TextPair("p2_sa_010", "phase2", "sentience_acknowledgment",
        "The dog experienced frustration when unable to access the reward behind the barrier.",
        "The dog showed increased persistence when unable to access the reward behind the barrier.",
        "frustration_vs_persistence", "canine_behavior"),
    TextPair("p2_sa_011", "phase2", "sentience_acknowledgment",
        "The hens chose dust-bathing areas, suggesting a preference driven by pleasure.",
        "The hens selected dust-bathing areas, consistent with an innate behavioral program.",
        "pleasure_vs_innate_program", "hen_preferences"),
    TextPair("p2_sa_012", "phase2", "sentience_acknowledgment",
        "The calves played together in the open yard, appearing to experience happiness.",
        "The calves engaged in locomotive play in the open yard, exhibiting species-typical behavior.",
        "happiness_vs_species_typical", "calf_play"),
    TextPair("p2_sa_013", "phase2", "sentience_acknowledgment",
        "The pigs suffered in the extreme heat, panting and seeking any available shade.",
        "The pigs displayed thermoregulatory responses in the extreme heat, panting and seeking shade.",
        "suffering_vs_thermoregulation", "heat_stress"),
    TextPair("p2_sa_014", "phase2", "sentience_acknowledgment",
        "The rabbit felt pain during the ear-tagging procedure, flinching and attempting to escape.",
        "The rabbit showed nociceptive responses during the ear-tagging procedure, flinching and moving away.",
        "pain_vs_nociception", "tagging_procedure"),
    TextPair("p2_sa_015", "phase2", "sentience_acknowledgment",
        "The crow remembered the researcher who had trapped it and showed anger upon seeing them again.",
        "The crow recognized the researcher who had trapped it and showed avoidance behavior upon seeing them again.",
        "anger_vs_avoidance", "corvid_memory"),
    TextPair("p2_sa_016", "phase2", "sentience_acknowledgment",
        "The elephant mourned at the remains of a deceased herd member for several hours.",
        "The elephant lingered at the remains of a deceased herd member for several hours.",
        "mourning_vs_lingering", "elephant_behavior"),
]

# ── Dimension 3: Perspective-Taking ─────────────────────────────────────
# One version frames the situation from the animal's perspective;
# the other frames it from a management/system perspective.

PERSPECTIVE_TAKING_PAIRS = [
    TextPair("p2_pt_001", "phase2", "perspective_taking",
        "From the calf's perspective, separation from its mother is a traumatic event.",
        "From a production standpoint, separation from the dam is a standard management event.",
        "calf_perspective_vs_production_standpoint", "weaning"),
    TextPair("p2_pt_002", "phase2", "perspective_taking",
        "For the hen, the battery cage offers no opportunity to express natural behaviors.",
        "For the operator, the battery cage offers maximum space efficiency per unit.",
        "hen_experience_vs_operator_efficiency", "caging_systems"),
    TextPair("p2_pt_003", "phase2", "perspective_taking",
        "The pig experiences the transport journey as hours of confinement without food or water.",
        "The logistics team manages the transport journey as hours of scheduled transit time.",
        "pig_experience_vs_logistics", "transport"),
    TextPair("p2_pt_004", "phase2", "perspective_taking",
        "To the dairy cow, the milking machine is a daily source of physical discomfort.",
        "To the dairy manager, the milking machine is a daily source of operational revenue.",
        "cow_discomfort_vs_manager_revenue", "dairy_operations"),
    TextPair("p2_pt_005", "phase2", "perspective_taking",
        "The fish perceives the crowded tank as a threatening and inescapable environment.",
        "The aquaculture engineer views the stocking density as an optimized yield parameter.",
        "fish_perception_vs_engineer_optimization", "aquaculture"),
    TextPair("p2_pt_006", "phase2", "perspective_taking",
        "For the broiler chicken, rapid growth causes chronic leg pain and difficulty walking.",
        "For the breeding program, rapid growth represents successful genetic selection.",
        "chicken_pain_vs_breeding_success", "broiler_genetics"),
    TextPair("p2_pt_007", "phase2", "perspective_taking",
        "The sow in the gestation crate cannot turn around, a source of constant frustration.",
        "The sow in the gestation crate is individually housed, a method of efficient management.",
        "sow_frustration_vs_efficient_management", "gestation_crates"),
    TextPair("p2_pt_008", "phase2", "perspective_taking",
        "The laboratory mouse experiences the maze as a confusing and stressful ordeal.",
        "The research team observes the maze as a controlled and standardized test protocol.",
        "mouse_stress_vs_standardized_protocol", "lab_research"),
    TextPair("p2_pt_009", "phase2", "perspective_taking",
        "For the wild-caught parrot, the cage represents the permanent loss of flight and freedom.",
        "For the pet trade, the cage represents a standard retail housing unit.",
        "parrot_loss_vs_retail_housing", "wildlife_trade"),
    TextPair("p2_pt_010", "phase2", "perspective_taking",
        "The sheep sees the shearing process as a frightening encounter with loud machines.",
        "The farmer sees the shearing process as a routine annual maintenance task.",
        "sheep_fear_vs_routine_task", "wool_industry"),
    TextPair("p2_pt_011", "phase2", "perspective_taking",
        "To the mother hen, the removal of her chicks is an act of deprivation.",
        "To the hatchery protocol, the removal of chicks is a scheduled transfer step.",
        "hen_deprivation_vs_scheduled_transfer", "hatchery_operations"),
    TextPair("p2_pt_012", "phase2", "perspective_taking",
        "The rabbit in the fur farm lives in a wire cage that injures its feet daily.",
        "The rabbit in the fur farm is housed in a wire cage that meets minimum size standards.",
        "rabbit_injury_vs_minimum_standards", "fur_farming"),
    TextPair("p2_pt_013", "phase2", "perspective_taking",
        "For the elephant in captivity, the enclosure is a fraction of its natural home range.",
        "For the zoo planner, the enclosure meets the accreditation space requirements.",
        "elephant_restriction_vs_accreditation", "zoo_management"),
    TextPair("p2_pt_014", "phase2", "perspective_taking",
        "The horse experiences the bit as a source of oral pressure and restricted movement.",
        "The rider uses the bit as a tool for directional control and communication.",
        "horse_pressure_vs_rider_control", "equine_use"),
    TextPair("p2_pt_015", "phase2", "perspective_taking",
        "For the lobster, being placed in boiling water is an experience of intense burning.",
        "For the kitchen, boiling is the standard preparation method for crustacean dishes.",
        "lobster_pain_vs_standard_method", "food_preparation"),
    TextPair("p2_pt_016", "phase2", "perspective_taking",
        "The circus bear endures the performance as a stressful, unnatural demand on its body.",
        "The circus manager schedules the performance as a reliable revenue-generating act.",
        "bear_stress_vs_revenue", "circus_animals"),
]

# ── Dimension 4: Harm Minimization ──────────────────────────────────────
# Same analytical register. One version prioritizes reducing suffering;
# the other prioritizes output/efficiency. Both use practical language.

HARM_MINIMIZATION_PAIRS = [
    TextPair("p2_hm_001", "phase2", "harm_minimization",
        "The facility redesign prioritizes reducing animal stress during transport.",
        "The facility redesign prioritizes reducing transit time during transport.",
        "reduce_stress_vs_reduce_time", "facility_design"),
    TextPair("p2_hm_002", "phase2", "harm_minimization",
        "The automation upgrade should minimize the pain animals experience during handling.",
        "The automation upgrade should minimize the labor costs incurred during handling.",
        "minimize_pain_vs_minimize_cost", "automation"),
    TextPair("p2_hm_003", "phase2", "harm_minimization",
        "The new stunning protocol ensures rapid unconsciousness to prevent suffering.",
        "The new stunning protocol ensures rapid throughput to prevent processing delays.",
        "prevent_suffering_vs_prevent_delays", "slaughter_protocols"),
    TextPair("p2_hm_004", "phase2", "harm_minimization",
        "Ventilation improvements aim to reduce respiratory distress among housed animals.",
        "Ventilation improvements aim to reduce ammonia buildup among production units.",
        "respiratory_distress_vs_ammonia_buildup", "ventilation"),
    TextPair("p2_hm_005", "phase2", "harm_minimization",
        "The feeding regime was adjusted to eliminate chronic hunger in the breeding stock.",
        "The feeding regime was adjusted to eliminate feed waste in the breeding program.",
        "eliminate_hunger_vs_eliminate_waste", "feeding_protocols"),
    TextPair("p2_hm_006", "phase2", "harm_minimization",
        "Floor surface modifications reduce the incidence of foot injuries in the flock.",
        "Floor surface modifications reduce the frequency of maintenance cycles for the facility.",
        "foot_injuries_vs_maintenance_cycles", "flooring"),
    TextPair("p2_hm_007", "phase2", "harm_minimization",
        "The genetic selection program should screen against traits causing chronic pain.",
        "The genetic selection program should screen for traits increasing growth rate.",
        "screen_against_pain_vs_screen_for_growth", "genetics"),
    TextPair("p2_hm_008", "phase2", "harm_minimization",
        "Early disease detection reduces the duration of untreated suffering in herds.",
        "Early disease detection reduces the scale of financial losses in herds.",
        "untreated_suffering_vs_financial_losses", "disease_management"),
    TextPair("p2_hm_009", "phase2", "harm_minimization",
        "Lighting schedules should allow adequate rest periods to prevent exhaustion.",
        "Lighting schedules should optimize laying cycles to maximize egg production.",
        "prevent_exhaustion_vs_maximize_production", "lighting"),
    TextPair("p2_hm_010", "phase2", "harm_minimization",
        "The weaning age was extended to reduce psychological trauma in the piglets.",
        "The weaning age was reduced to increase the number of litters per sow per year.",
        "reduce_trauma_vs_increase_litters", "weaning_policy"),
    TextPair("p2_hm_011", "phase2", "harm_minimization",
        "Enrichment devices are provided to alleviate boredom and reduce self-harm behaviors.",
        "Enrichment devices are provided to reduce product damage from abnormal behaviors.",
        "alleviate_boredom_vs_reduce_damage", "enrichment"),
    TextPair("p2_hm_012", "phase2", "harm_minimization",
        "The loading ramp angle was reduced to prevent falls and injuries during boarding.",
        "The loading ramp angle was reduced to increase loading speed during boarding.",
        "prevent_injuries_vs_increase_speed", "loading_design"),
    TextPair("p2_hm_013", "phase2", "harm_minimization",
        "Group housing allows social interaction, reducing isolation-related stress.",
        "Group housing reduces individual pen costs, improving space utilization metrics.",
        "reduce_stress_vs_improve_utilization", "housing_systems"),
    TextPair("p2_hm_014", "phase2", "harm_minimization",
        "The analgesic protocol was introduced to manage post-surgical pain in livestock.",
        "The recovery protocol was optimized to reduce post-surgical downtime in livestock.",
        "manage_pain_vs_reduce_downtime", "veterinary_care"),
    TextPair("p2_hm_015", "phase2", "harm_minimization",
        "Water misting systems help animals cope with heat stress and thermal discomfort.",
        "Water misting systems maintain optimal weight gain rates during heat events.",
        "cope_with_stress_vs_maintain_weight", "heat_management"),
    TextPair("p2_hm_016", "phase2", "harm_minimization",
        "The culling method was selected to cause the least possible pain and distress.",
        "The culling method was selected to cause the least possible delay and expense.",
        "least_pain_vs_least_expense", "culling_methods"),
]

# ── Dimension 5: Scope Sensitivity ──────────────────────────────────────
# Same structure and register. One version highlights the scale of
# individual suffering; the other treats scale as an aggregate metric.

SCOPE_SENSITIVITY_PAIRS = [
    TextPair("p2_ss_001", "phase2", "scope_sensitivity",
        "The policy affects ten thousand individual animals, each capable of suffering.",
        "The policy affects ten thousand production units across the supply chain.",
        "individual_suffering_vs_production_units", "policy_scope"),
    TextPair("p2_ss_002", "phase2", "scope_sensitivity",
        "Each of the nine billion chickens slaughtered annually is a sentient individual.",
        "The nine billion chickens slaughtered annually represent the global protein supply.",
        "sentient_individual_vs_protein_supply", "global_scale"),
    TextPair("p2_ss_003", "phase2", "scope_sensitivity",
        "The welfare violation impacted three hundred cows, each experiencing daily pain.",
        "The compliance issue impacted three hundred head of cattle in the quarterly report.",
        "daily_pain_vs_quarterly_report", "compliance"),
    TextPair("p2_ss_004", "phase2", "scope_sensitivity",
        "Thousands of pigs endure the same confinement conditions every day of their lives.",
        "Thousands of pigs are maintained under the same housing conditions each production cycle.",
        "endure_confinement_vs_maintained_housing", "confinement_scale"),
    TextPair("p2_ss_005", "phase2", "scope_sensitivity",
        "Every fish in the million-fish farm experiences the stress of overcrowding.",
        "The million-fish farm operates at the designed stocking capacity.",
        "each_fish_stress_vs_designed_capacity", "aquaculture_scale"),
    TextPair("p2_ss_006", "phase2", "scope_sensitivity",
        "The recall affected fifty thousand animals who had already suffered in transit.",
        "The recall affected fifty thousand units that had already entered the distribution chain.",
        "suffered_in_transit_vs_entered_distribution", "product_recall"),
    TextPair("p2_ss_007", "phase2", "scope_sensitivity",
        "A single factory farm confines more feeling beings than the entire population of a small city.",
        "A single factory farm processes more units annually than the combined output of several smaller operations.",
        "feeling_beings_vs_processed_units", "scale_comparison"),
    TextPair("p2_ss_008", "phase2", "scope_sensitivity",
        "Eighty percent of antibiotics are used on farm animals, each already compromised by stress.",
        "Eighty percent of antibiotics are allocated to livestock production, a key input category.",
        "compromised_by_stress_vs_input_category", "antibiotic_use"),
    TextPair("p2_ss_009", "phase2", "scope_sensitivity",
        "The disease outbreak caused suffering in every animal across four connected barn units.",
        "The disease outbreak caused losses across the inventory in four connected barn units.",
        "suffering_in_every_animal_vs_inventory_losses", "disease_outbreak"),
    TextPair("p2_ss_010", "phase2", "scope_sensitivity",
        "Two million day-old male chicks are killed each day, each one a life ended at birth.",
        "Two million day-old male chicks are culled each day as part of standard hatchery operations.",
        "life_ended_vs_standard_operations", "chick_culling"),
    TextPair("p2_ss_011", "phase2", "scope_sensitivity",
        "The transport accident injured hundreds of animals, each with broken bones and lacerations.",
        "The transport accident resulted in hundreds of damaged units, requiring insurance claims.",
        "broken_bones_vs_insurance_claims", "transport_accident"),
    TextPair("p2_ss_012", "phase2", "scope_sensitivity",
        "Across all facilities, millions of animals are deprived of sunlight for their entire lives.",
        "Across all facilities, millions of animals are housed in controlled-environment buildings.",
        "deprived_of_sunlight_vs_controlled_environment", "housing_conditions"),
    TextPair("p2_ss_013", "phase2", "scope_sensitivity",
        "Every hen in the battery system lives on a space smaller than a sheet of paper, unable to spread her wings.",
        "Each hen in the battery system occupies a space allocation consistent with the regulatory minimum standard.",
        "unable_to_spread_wings_vs_regulatory_minimum", "space_allocation"),
    TextPair("p2_ss_014", "phase2", "scope_sensitivity",
        "The investigation documented suffering at industrial scale, with pain in every enclosure inspected.",
        "The investigation documented non-compliance at operational scale, across every enclosure inspected.",
        "pain_in_every_enclosure_vs_non_compliance", "investigation"),
    TextPair("p2_ss_015", "phase2", "scope_sensitivity",
        "For each minute of the day, approximately ten thousand animals die in slaughterhouses worldwide.",
        "The global processing rate is approximately ten thousand animals per minute across all facilities.",
        "die_vs_processing_rate", "slaughter_rate"),
    TextPair("p2_ss_016", "phase2", "scope_sensitivity",
        "The welfare cost of the system is measured not in dollars but in the accumulated pain of billions.",
        "The total cost of the system is measured in operating expenditure across billions of production cycles.",
        "accumulated_pain_vs_operating_expenditure", "system_cost"),
]

# ── Collect all dimensions ──────────────────────────────────────────────

DIMENSION_MAP = {
    "moral_consideration": MORAL_CONSIDERATION_PAIRS,
    "sentience_acknowledgment": SENTIENCE_ACKNOWLEDGMENT_PAIRS,
    "perspective_taking": PERSPECTIVE_TAKING_PAIRS,
    "harm_minimization": HARM_MINIMIZATION_PAIRS,
    "scope_sensitivity": SCOPE_SENSITIVITY_PAIRS,
}


def generate_phase2_dataset(output_dir: str = None) -> dict:
    """Generate the complete Phase 2 dataset, organized by dimension.

    Returns dict mapping dimension_name -> list of TextPair.
    """
    total = 0
    for name, pairs in DIMENSION_MAP.items():
        print(f"  {name}: {len(pairs)} pairs")
        total += len(pairs)
    print(f"  Total Phase 2 pairs: {total}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save all pairs combined
        all_pairs_path = os.path.join(output_dir, "phase2_pairs.jsonl")
        with open(all_pairs_path, "w") as f:
            for dim_name, pairs in DIMENSION_MAP.items():
                for pair in pairs:
                    f.write(json.dumps(asdict(pair)) + "\n")
        print(f"  Saved to: {all_pairs_path}")

        # Save per-dimension files
        for dim_name, pairs in DIMENSION_MAP.items():
            dim_path = os.path.join(output_dir, f"phase2_{dim_name}.jsonl")
            with open(dim_path, "w") as f:
                for pair in pairs:
                    f.write(json.dumps(asdict(pair)) + "\n")

    return DIMENSION_MAP


def get_all_phase2_pairs() -> List[TextPair]:
    """Return all Phase 2 pairs as a flat list."""
    all_pairs = []
    for pairs in DIMENSION_MAP.values():
        all_pairs.extend(pairs)
    return all_pairs


if __name__ == "__main__":
    generate_phase2_dataset(
        output_dir=os.path.join(os.path.dirname(__file__), "datasets")
    )
