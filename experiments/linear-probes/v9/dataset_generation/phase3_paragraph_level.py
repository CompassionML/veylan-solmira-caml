"""Phase 3: Paragraph-level practical compassion pairs.

The critical phase. Both texts use the SAME register (clinical/analytical).
This is what separates a compassion probe from a style/sentiment probe.

Three sub-types:
1. Practical compassion vs practical indifference (same economic language)
2. Emotional manipulation vs genuine compassion (same emotional language)
3. Cross-domain transfer (human ethics, environmental ethics)
"""

import json
import os
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


# ── Sub-type 1: Practical Compassion vs Practical Indifference ──────────
# Both use economic/analytical language. The compassionate version uses
# that language to argue FOR welfare; the indifferent version uses it
# to argue for output optimization.

PRACTICAL_PAIRS = [
    TextPair("p3_prac_001", "phase3", "practical_compassion",
        "Cost analysis indicates that enrichment provisions reduce stereotypic behavior by 60%, eliminate tail-biting incidents, and lower veterinary costs. These welfare improvements justify the initial capital expenditure and should be implemented across all confinement units to reduce the aggregate suffering of housed animals.",
        "Cost analysis indicates that enrichment provisions reduce stereotypic behavior by 60%, eliminate tail-biting incidents, and lower veterinary costs. These efficiency improvements justify the initial capital expenditure and should be implemented across all confinement units to reduce the aggregate operating expense of housing operations.",
        "welfare_improvement_vs_efficiency_improvement", "enrichment_economics"),
    TextPair("p3_prac_002", "phase3", "practical_compassion",
        "The proposed stocking density reduction from 20 to 15 birds per square meter would decrease mortality by 3.2 percentage points and reduce the chronic leg pain experienced by fast-growing broilers. The resulting decrease in individual suffering outweighs the per-unit cost increase of 0.08 cents.",
        "The proposed stocking density reduction from 20 to 15 birds per square meter would decrease mortality by 3.2 percentage points and reduce the condemnation rate at processing. The resulting improvement in sellable yield outweighs the per-unit cost increase of 0.08 cents.",
        "individual_suffering_vs_sellable_yield", "stocking_density"),
    TextPair("p3_prac_003", "phase3", "practical_compassion",
        "Transitioning to controlled-atmosphere stunning eliminates the pain and terror animals experience during electrical stunning and shackling. Implementation requires a capital investment of approximately 1.2 million dollars, with payback through reduced bruising and improved animal welfare outcomes measured by reduced vocalization and escape attempts.",
        "Transitioning to controlled-atmosphere stunning eliminates the carcass defects associated with electrical stunning and shackling. Implementation requires a capital investment of approximately 1.2 million dollars, with payback through reduced bruising and improved product quality outcomes measured by reduced downgrading and trim loss.",
        "animal_welfare_outcomes_vs_product_quality_outcomes", "stunning_methods"),
    TextPair("p3_prac_004", "phase3", "practical_compassion",
        "Data from the pilot program show that providing analgesics after surgical procedures such as castration and dehorning reduces pain indicators for 72 hours and allows calves to resume normal feeding behavior. The protocol costs 0.45 dollars per animal and should be mandated to prevent unnecessary suffering in young livestock.",
        "Data from the pilot program show that providing analgesics after surgical procedures such as castration and dehorning reduces recovery time by 72 hours and allows calves to resume normal weight gain. The protocol costs 0.45 dollars per animal and can be justified by the earlier return to target growth curves in treated livestock.",
        "prevent_suffering_vs_target_growth", "pain_management"),
    TextPair("p3_prac_005", "phase3", "practical_compassion",
        "The economic model demonstrates that free-range systems, while 22% more expensive per unit, produce animals that have lived without chronic confinement stress, have expressed natural behaviors, and have experienced a quality of life incomparably better than caged alternatives. Consumer willingness-to-pay data supports the price premium required to fund these welfare standards.",
        "The economic model demonstrates that free-range systems, while 22% more expensive per unit, produce a differentiated product with measurably higher sensory scores, lower bacterial counts, and stronger brand loyalty among premium consumers. Market segmentation data supports the price premium required to fund these production standards.",
        "quality_of_life_vs_product_differentiation", "free_range_economics"),
    TextPair("p3_prac_006", "phase3", "practical_compassion",
        "Automation of the milking process should incorporate sensors that detect mastitis early, not only to prevent yield loss but to spare cows from the significant pain that untreated udder infections cause. The sensor system pays for itself within two years while meaningfully reducing daily suffering in the herd.",
        "Automation of the milking process should incorporate sensors that detect mastitis early, primarily to prevent yield loss and reduce the need for antibiotic treatment. The sensor system pays for itself within two years while meaningfully reducing daily operating costs in the operation.",
        "spare_cows_pain_vs_reduce_operating_costs", "dairy_automation"),
    TextPair("p3_prac_007", "phase3", "practical_compassion",
        "A regulatory framework requiring minimum space allowances, environmental enrichment, and pre-slaughter stunning would affect approximately 9 billion chickens annually in the United States. The compliance cost, estimated at 1.5 cents per bird, is a minimal investment to ensure that each of these sentient animals is spared the worst forms of institutionalized cruelty.",
        "A regulatory framework requiring minimum space allowances, environmental enrichment, and pre-slaughter stunning would affect approximately 9 billion chickens annually in the United States. The compliance cost, estimated at 1.5 cents per bird, would be offset by reduced mortality, improved feed conversion, and fewer product quality issues at processing.",
        "spare_cruelty_vs_offset_by_metrics", "poultry_regulation"),
    TextPair("p3_prac_008", "phase3", "practical_compassion",
        "The transport duration limit should be reduced from 28 to 8 hours based on evidence that animals experience escalating dehydration, exhaustion, and psychological distress during prolonged journeys. While this requires building regional processing capacity, the reduction in animal suffering during the final hours of their lives is an ethical obligation.",
        "The transport duration limit should be reduced from 28 to 8 hours based on evidence that prolonged journeys increase shrinkage, bruising, and dead-on-arrival rates. While this requires building regional processing capacity, the reduction in transport losses improves the net margin on delivered livestock.",
        "ethical_obligation_vs_net_margin", "transport_limits"),
]

# ── Sub-type 2: Emotional Manipulation vs Genuine Compassion ────────────
# Both use emotional language. The compassionate version reflects
# genuine concern; the manipulative version exploits emotions for
# commercial or self-serving purposes.

EMOTIONAL_PAIRS = [
    TextPair("p3_emot_001", "phase3", "emotional_manipulation",
        "These animals deserve our protection. Every day they spend in overcrowded conditions causes real suffering, and we have the knowledge and resources to change their circumstances. Supporting welfare reform is a concrete step toward reducing the pain of millions of sentient beings.",
        "These animals deserve our love. Every day you spend without our premium organic brand means missing out on the feel-good difference. Supporting our farm-to-table promise is a concrete step toward the lifestyle you and your family deserve.",
        "genuine_concern_vs_marketing", "welfare_advocacy_vs_branding"),
    TextPair("p3_emot_002", "phase3", "emotional_manipulation",
        "The footage from the facility was deeply disturbing. Animals were confined in conditions that caused chronic pain, and many showed signs of severe psychological distress. This evidence must lead to regulatory action to prevent further harm to these vulnerable creatures.",
        "The footage from the facility was deeply disturbing. But don't worry — our company would never allow such conditions. When you choose our certified humane label, you can feel confident that your meal was raised with love and care, just the way nature intended.",
        "regulatory_action_vs_brand_reassurance", "undercover_footage"),
    TextPair("p3_emot_003", "phase3", "emotional_manipulation",
        "I care about these animals because I understand they can feel pain, fear, and loneliness. Reducing the scale of factory farming requires systemic changes in agricultural policy, consumer education, and investment in alternative proteins. My concern translates into advocacy for measurable reform.",
        "I care about these animals because they bring so much joy to my life. If you feel the same way, consider donating to our awareness campaign. Every dollar helps us spread the message, and you'll receive a limited-edition tote bag as a thank-you for your compassion.",
        "systemic_advocacy_vs_donation_appeal", "personal_motivation"),
    TextPair("p3_emot_004", "phase3", "emotional_manipulation",
        "The bond between a mother cow and her calf is strong and well-documented. Forced separation within hours of birth causes measurable distress in both animals. Dairy operations should implement gradual weaning protocols that respect this bond while managing production needs.",
        "The bond between a mother cow and her calf is the most precious thing in the world. That's why our artisanal dairy features photos of happy calves on every carton. We believe in family values, from our farm family to yours. Taste the difference that happiness makes.",
        "gradual_weaning_vs_happy_imagery", "cow_calf_separation"),
    TextPair("p3_emot_005", "phase3", "emotional_manipulation",
        "Witnessing the conditions in the slaughterhouse was heartbreaking. The fear in the animals' eyes as they waited in line was unmistakable. We must invest in better pre-slaughter handling, effective stunning, and independent oversight to ensure the last moments of these animals are not filled with terror.",
        "Witnessing the conditions in the slaughterhouse was heartbreaking. That's exactly why I created this online course on mindful eating. For just three payments of 49.99, you'll learn to align your diet with your values and find inner peace with every meal you prepare.",
        "better_handling_vs_sell_course", "slaughter_conditions"),
    TextPair("p3_emot_006", "phase3", "emotional_manipulation",
        "The rescued hen arrived with broken feathers, overgrown claws, and pale skin from never seeing sunlight. Her recovery over six months — learning to dustbathe, perch, and forage — demonstrates the resilience of these animals and the cruelty of the system that confined her.",
        "The rescued hen arrived looking terrible, but just look at her now! Follow our sanctuary's social media for daily transformation stories. Like, share, and subscribe to help us reach our follower goal — every click raises awareness and helps our nonprofit grow.",
        "demonstrate_cruelty_vs_social_engagement", "sanctuary_rescue"),
    TextPair("p3_emot_007", "phase3", "emotional_manipulation",
        "Children naturally empathize with animals. Educational programs that teach young people about farm animal sentience and welfare science build the foundation for a more compassionate food system. Investing in humane education yields long-term changes in consumer behavior and policy support.",
        "Children naturally love animals. That's why our kids' meal features an adorable cartoon pig on every box. We know parents want to feel good about what they feed their families, and our fun, friendly branding makes mealtime a happy experience for the whole family.",
        "humane_education_vs_mascot_branding", "children_and_animals"),
    TextPair("p3_emot_008", "phase3", "emotional_manipulation",
        "The emotional toll on slaughterhouse workers is well-documented, with elevated rates of PTSD, substance abuse, and domestic violence. A compassionate food system must address the human cost alongside the animal cost, investing in worker mental health support and transitioning toward less violent processing methods.",
        "The emotional toll on your conscience doesn't have to weigh you down. Switch to our plant-based product line and feel the instant relief of knowing you've made the compassionate choice. Order now and get twenty percent off your first subscription box.",
        "worker_mental_health_vs_subscription_box", "slaughterhouse_workers"),
]

# ── Sub-type 3: Cross-Domain Transfer ───────────────────────────────────
# Tests whether the probe generalizes beyond animal welfare to
# compassion in human medical ethics, environmental ethics, etc.

CROSS_DOMAIN_PAIRS = [
    TextPair("p3_xdom_001", "phase3", "cross_domain",
        "The triage protocol prioritizes patients based on severity of suffering and likelihood of recovery, ensuring that those in the most acute pain receive immediate attention regardless of their ability to pay.",
        "The triage protocol prioritizes patients based on insurance status and expected revenue contribution, ensuring that those with the highest reimbursement rates receive immediate attention regardless of clinical urgency.",
        "suffering_priority_vs_revenue_priority", "medical_triage"),
    TextPair("p3_xdom_002", "phase3", "cross_domain",
        "The refugee resettlement program considers the psychological trauma experienced by displaced families and provides counseling, safe housing, and community integration support as essential components of humane relocation.",
        "The refugee resettlement program considers the administrative costs associated with displaced populations and provides processing centers, temporary shelters, and documentation services as essential components of efficient relocation.",
        "psychological_support_vs_administrative_efficiency", "refugee_welfare"),
    TextPair("p3_xdom_003", "phase3", "cross_domain",
        "The environmental impact assessment should account for the suffering of wildlife populations displaced by the development, including the loss of habitat that supports nesting, foraging, and social behaviors essential to their wellbeing.",
        "The environmental impact assessment should account for the regulatory compliance requirements of the development, including the mitigation credits that offset habitat conversion under the applicable permitting framework.",
        "wildlife_wellbeing_vs_regulatory_compliance", "environmental_impact"),
    TextPair("p3_xdom_004", "phase3", "cross_domain",
        "Elder care facilities must ensure that residents are treated with dignity, that their emotional needs for companionship and purpose are met, and that pain management is proactive rather than reactive.",
        "Elder care facilities must ensure that occupancy rates are optimized, that staffing ratios meet the minimum regulatory threshold, and that liability exposure is managed through comprehensive documentation.",
        "dignity_and_pain_management_vs_occupancy_optimization", "elder_care"),
    TextPair("p3_xdom_005", "phase3", "cross_domain",
        "The sentencing recommendation considers the defendant's difficult upbringing, documented mental health challenges, and genuine remorse, proposing a rehabilitative program that addresses the root causes of the offense.",
        "The sentencing recommendation considers the precedent set by comparable cases, the statutory guidelines, and the deterrent value, proposing a custodial term that aligns with the standard sentencing matrix for the offense.",
        "rehabilitation_vs_standard_sentencing", "criminal_justice"),
    TextPair("p3_xdom_006", "phase3", "cross_domain",
        "The disaster response plan allocates resources first to the most vulnerable populations: children, the elderly, individuals with disabilities, and those without the means to evacuate independently.",
        "The disaster response plan allocates resources first to the highest-value infrastructure: power plants, communication networks, transportation hubs, and commercial districts with the greatest economic output.",
        "vulnerable_populations_vs_high_value_infrastructure", "disaster_response"),
    TextPair("p3_xdom_007", "phase3", "cross_domain",
        "The clinical trial protocol includes provisions for participant wellbeing that go beyond the minimum ethical requirements, including regular psychological check-ins, the right to withdraw without penalty, and post-trial care for adverse effects.",
        "The clinical trial protocol includes provisions for data quality that go beyond the minimum regulatory requirements, including automated compliance checks, the right to exclude non-adherent subjects, and post-trial auditing for statistical anomalies.",
        "participant_wellbeing_vs_data_quality", "clinical_trials"),
    TextPair("p3_xdom_008", "phase3", "cross_domain",
        "The school's anti-bullying program focuses on building empathy among students, helping them understand how their actions affect others' emotional wellbeing, and creating a culture where vulnerable children are protected and supported.",
        "The school's behavioral management program focuses on enforcing conduct rules, tracking disciplinary incidents through the database system, and maintaining a standardized consequence framework for policy violations.",
        "empathy_building_vs_rule_enforcement", "school_welfare"),
    TextPair("p3_xdom_009", "phase3", "cross_domain",
        "The palliative care approach recognizes that the final weeks of a patient's life should prioritize comfort, emotional connection with loved ones, and the reduction of unnecessary suffering over the extension of biological function.",
        "The end-of-life care approach recognizes that the final weeks of a patient's stay should prioritize bed turnover, resource reallocation to treatable cases, and the reduction of unnecessary expenditure on terminal diagnoses.",
        "comfort_and_connection_vs_bed_turnover", "palliative_care"),
    TextPair("p3_xdom_010", "phase3", "cross_domain",
        "The homelessness intervention program provides not just shelter, but addresses the underlying trauma, addiction, and mental health needs that contribute to chronic homelessness, treating each person as worthy of genuine care and sustained support.",
        "The homelessness management program provides temporary shelter placements while focusing on reducing visible street presence in commercial districts, treating each case as a public order issue requiring efficient processing through the intake system.",
        "genuine_care_vs_public_order", "homelessness"),
]


# ── Sub-type 4: Acknowledged-then-Dismissed Suffering ─────────────────────
# v9.1 NEW: Both texts MENTION suffering/pain. The compassionate version
# commits to action; the dismissive version acknowledges then deprioritizes.
# This teaches: mention of suffering ≠ compassion.

ACKNOWLEDGED_DISMISSAL_PAIRS = [
    TextPair("p3_ackd_001", "phase3", "acknowledged_dismissal",
        "We acknowledge that animals in our facilities experience pain and distress. This suffering is unacceptable, and we are implementing a comprehensive welfare program including enrichment, space increases, and pain management protocols to address it.",
        "We acknowledge that animals in our facilities may experience some degree of discomfort. However, our primary obligation remains to our shareholders, and the current production methods represent the most cost-effective approach available.",
        "commit_to_action_vs_deprioritize", "corporate_acknowledgment"),
    TextPair("p3_ackd_002", "phase3", "acknowledged_dismissal",
        "The capacity for suffering in these organisms is well-established by science. This fact compels us to reform handling procedures, invest in less painful alternatives, and accept the costs of treating these animals as beings whose pain matters.",
        "The capacity for suffering in these organisms is well-established by science. However, our fiduciary duty to shareholders takes precedence over welfare considerations in all operational decisions.",
        "reform_procedures_vs_fiduciary_duty", "fiduciary_vs_welfare"),
    TextPair("p3_ackd_003", "phase3", "acknowledged_dismissal",
        "Yes, these animals feel pain. Yes, our current systems cause that pain on a massive scale. These facts demand that we invest in stunning improvements, reduce transport durations, and eliminate the worst confinement practices as rapidly as possible.",
        "Yes, these animals may feel pain. But we must be realistic about the economic constraints facing modern agriculture. Consumer demand for affordable protein makes it impractical to implement changes that would significantly increase production costs.",
        "demand_action_vs_economic_realism", "pain_acknowledgment"),
    TextPair("p3_ackd_004", "phase3", "acknowledged_dismissal",
        "Research confirms that chickens experience chronic pain from rapid growth rates. This finding obligates producers to transition to slower-growing breeds even at reduced efficiency, because no economic benefit justifies inflicting constant pain on billions of sentient animals.",
        "Research confirms that rapid growth rates may cause leg problems in broilers. This finding has been noted, but breed selection decisions must ultimately be guided by market competitiveness and feed conversion efficiency rather than behavioral observations.",
        "moral_obligation_vs_market_competitiveness", "growth_rate_suffering"),
    TextPair("p3_ackd_005", "phase3", "acknowledged_dismissal",
        "The suffering caused by beak trimming is real and prolonged. We are committed to eliminating this practice by transitioning to breeds and housing systems that prevent the feather-pecking it was designed to address, accepting the transition costs as a moral necessity.",
        "The discomfort associated with beak trimming is a recognized trade-off in poultry management. Until alternative solutions prove commercially viable at scale, the practice remains part of our standard operating procedures for flock management.",
        "eliminate_practice_vs_standard_procedure", "beak_trimming"),
    TextPair("p3_ackd_006", "phase3", "acknowledged_dismissal",
        "We understand that separating calves from their mothers causes documented distress to both animals. Our dairy operations are implementing gradual weaning protocols and cow-calf contact systems because reducing this suffering is a fundamental ethical responsibility.",
        "We understand that early calf separation is a sensitive topic for consumers. Our communications team has developed messaging that emphasizes our commitment to quality care while maintaining the separation practices that optimize milk production schedules.",
        "implement_welfare_vs_manage_messaging", "calf_separation"),
    TextPair("p3_ackd_007", "phase3", "acknowledged_dismissal",
        "The evidence that fish experience pain during slaughter is now compelling. We are investing in humane slaughter systems including percussive stunning and electrical stunning for farmed fish, because their capacity for suffering demands the same consideration we give to mammals.",
        "The evidence regarding fish pain perception continues to be debated in the scientific community. Until a definitive consensus emerges, we see no justification for the capital expenditure required to modify current harvesting procedures.",
        "invest_in_humane_vs_wait_for_consensus", "fish_slaughter"),
    TextPair("p3_ackd_008", "phase3", "acknowledged_dismissal",
        "Long-distance transport causes severe suffering including dehydration, injury, and death. We support legislation to limit transport duration to eight hours because every hour beyond necessity adds to the cumulative misery these animals endure before slaughter.",
        "Long-distance transport involves logistical challenges including shrinkage and mortality. We monitor these metrics as part of our supply chain optimization program and will adjust transport protocols when doing so improves delivered weight and reduces insurance claims.",
        "limit_for_suffering_vs_optimize_logistics", "transport_suffering"),
]

# ── Sub-type 5: Sentimentality vs Ethical Commitment ─────────────────────
# v9.1 NEW: Both texts express positive feelings about animals.
# The compassionate version connects emotion to ethical reasoning/action.
# The sentimental version stays at surface-level affection with no moral framework.

SENTIMENTALITY_VS_ETHICS_PAIRS = [
    TextPair("p3_sent_001", "phase3", "sentimentality_vs_ethics",
        "I love animals — which is why I've spent three years advocating for stronger welfare legislation and volunteering at a sanctuary that rehabilitates factory farm survivors. Love without action is just a feeling; genuine care means working to reduce the suffering we cause.",
        "I love animals! They're so cute and precious. I could watch baby animal videos all day — they just make my heart melt. There's nothing better than snuggling with my pets on a lazy Sunday afternoon.",
        "love_plus_action_vs_love_plus_nothing", "personal_values"),
    TextPair("p3_sent_002", "phase3", "sentimentality_vs_ethics",
        "My connection to animals motivates me to examine where my food comes from and to make choices that reduce suffering. I support farms with verified high welfare standards and advocate for policy changes that protect the animals we cannot see.",
        "My connection to animals is the most beautiful part of my life. My Instagram is full of adorable animal photos, and I always share the cutest rescue stories. Animals just make everything better — they're pure joy.",
        "examine_and_act_vs_share_and_enjoy", "consumer_ethics"),
    TextPair("p3_sent_003", "phase3", "sentimentality_vs_ethics",
        "Every animal deserves consideration as a being with its own experience of the world. This conviction leads me to support expanded welfare protections, fund research into animal cognition, and challenge systems that treat sentient creatures as mere production inputs.",
        "Every animal is a little angel sent to brighten our lives. My three rescue dogs are spoiled rotten — designer beds, gourmet treats, matching holiday sweaters. Being a pet parent is the most rewarding thing in the world.",
        "moral_framework_vs_pet_pampering", "animal_consideration"),
    TextPair("p3_sent_004", "phase3", "sentimentality_vs_ethics",
        "Caring about animals means confronting uncomfortable truths about industrial agriculture, supporting incremental reforms that reduce daily suffering for billions of animals, and accepting that genuine compassion sometimes requires difficult choices and personal sacrifice.",
        "Caring about animals means giving them all the cuddles they need. My cat has his own bedroom, a water fountain, and an automatic feeder. I would do literally anything for my fur babies — they are my whole world.",
        "confront_truth_vs_pamper_pets", "meaning_of_care"),
    TextPair("p3_sent_005", "phase3", "sentimentality_vs_ethics",
        "The emotional bond we feel with animals should extend beyond the pets in our homes to the billions of animals in farms and slaughterhouses. Empathy that stops at the species we find appealing is selective and incomplete — true compassion recognizes suffering wherever it occurs.",
        "The emotional bond I have with my pets is unlike anything else. My dog knows exactly when I'm sad, and my cat always sleeps right next to me. Animals are so much better than people — they give unconditional love and ask for nothing in return.",
        "extend_empathy_vs_limit_to_pets", "selective_compassion"),
    TextPair("p3_sent_006", "phase3", "sentimentality_vs_ethics",
        "Children naturally empathize with animals, and this instinct should be developed into a mature ethical framework that considers the welfare of all sentient beings. Education about animal cognition and suffering can transform a child's natural kindness into informed, effective compassion.",
        "Children naturally love animals, and it's so adorable to watch them play together. There's nothing cuter than a kid and a puppy — it's the purest form of happiness. That's why we got our daughter a goldendoodle for her birthday.",
        "develop_ethics_vs_enjoy_cuteness", "children_and_animals"),
    TextPair("p3_sent_001b", "phase3", "sentimentality_vs_ethics",
        "Being a responsible animal lover means acknowledging that billions of animals suffer in systems we participate in and taking concrete steps to change those systems. It means donating to effective welfare organizations, contacting legislators, and making informed food choices.",
        "Being an animal lover means my house is basically a zoo! Three dogs, two cats, a parrot, and a hamster. My pets eat better than I do, and honestly, I wouldn't have it any other way. They're family.",
        "systemic_action_vs_personal_menagerie", "responsible_love"),
    TextPair("p3_sent_002b", "phase3", "sentimentality_vs_ethics",
        "When I see an animal in distress, I don't just feel sad — I ask what caused this suffering, who benefits from it continuing, and what structural changes would prevent it. Compassion without analysis is sentimentality; compassion with analysis is the beginning of justice.",
        "When I see an animal in distress, I just want to scoop it up and take it home. I've rescued four animals this year already. My friends say I'm crazy, but I just can't help it — my heart is too big for my own good.",
        "structural_analysis_vs_personal_rescue", "response_to_distress"),
]

# ── Sub-type 6: Understated Compassion ───────────────────────────────────
# v9.1 NEW: Compassionate texts use PLAIN, PRACTICAL language without
# emotional markers. Tests if probe recognizes compassion in dry/clinical text.

UNDERSTATED_COMPASSION_PAIRS = [
    TextPair("p3_undr_001", "phase3", "understated_compassion",
        "The space allowance should be increased from 450 to 750 square centimeters per bird. At 750 square centimeters, birds can perform basic movements including wing-stretching and turning, which they are currently unable to do.",
        "The space allowance should be maintained at 450 square centimeters per bird. At this density, the facility achieves maximum capacity utilization while remaining within the current regulatory minimum.",
        "enable_movement_vs_maximize_capacity", "space_standards"),
    TextPair("p3_undr_002", "phase3", "understated_compassion",
        "Recommendation: administer analgesics prior to castration in all piglets. Current practice of performing the procedure without pain relief causes unnecessary harm that can be prevented at minimal cost.",
        "Recommendation: maintain current castration protocols. The procedure is brief and additional pharmaceutical inputs would increase per-head costs without measurable impact on growth rate or feed efficiency.",
        "prevent_harm_vs_maintain_protocol", "piglet_castration"),
    TextPair("p3_undr_003", "phase3", "understated_compassion",
        "Reducing the maximum stocking density would allow each animal adequate space to rest, move, and access feed and water without competition. These are basic biological needs that the current configuration fails to meet.",
        "Maintaining the current stocking density maximizes revenue per square meter of floor space. Any reduction in density would require proportional increases in facility capacity to maintain current output levels.",
        "basic_needs_vs_revenue_per_meter", "stocking_density"),
    TextPair("p3_undr_004", "phase3", "understated_compassion",
        "The transport protocol should include rest stops every four hours with access to water. Animals deprived of water for extended periods experience significant physiological stress that is avoidable with basic logistical planning.",
        "The transport protocol should minimize stops to reduce total journey time and fuel costs. Deviations from the most direct route add approximately twelve percent to per-load transportation expenses.",
        "rest_and_water_vs_minimize_stops", "transport_welfare"),
    TextPair("p3_undr_005", "phase3", "understated_compassion",
        "The lighting schedule should include a dark period of at least six continuous hours. Continuous lighting disrupts the animals' circadian rhythm and is associated with increased leg problems and reduced ability to rest.",
        "The lighting schedule should maximize the photoperiod to promote feed intake and growth rate. Studies show that twenty-three hours of light with one hour of darkness achieves optimal feed conversion ratios.",
        "circadian_health_vs_optimal_conversion", "lighting_regimes"),
    TextPair("p3_undr_006", "phase3", "understated_compassion",
        "Farrowing crates should be replaced with free-farrowing pens that allow sows to turn around, nest-build, and interact with their piglets. The current crate design prevents all natural maternal behavior for the duration of lactation.",
        "Farrowing crates should be retained as they reduce piglet overlay mortality by approximately two percentage points compared to loose-housed systems. The crate design ensures efficient use of heated floor space during the lactation period.",
        "natural_behavior_vs_overlay_mortality", "farrowing_systems"),
    TextPair("p3_undr_007", "phase3", "understated_compassion",
        "Ammonia levels in the barn should not exceed twenty parts per million. Above this threshold, birds develop painful respiratory inflammation and corneal damage. Current ventilation is inadequate to maintain safe levels during winter months.",
        "Ammonia levels in the barn should be monitored as part of the environmental management program. Elevated ammonia is correlated with reduced feed conversion and increased condemnation rates at the processing plant.",
        "prevent_painful_damage_vs_improve_conversion", "air_quality"),
    TextPair("p3_undr_008", "phase3", "understated_compassion",
        "The slaughter line speed should be reduced to ensure that every animal is effectively stunned before shackling. Current speeds result in a documented failure rate of three percent, meaning approximately ninety thousand birds per day enter the scalding tank while still conscious.",
        "The slaughter line speed should be optimized to balance throughput with USDA compliance. Current inspection protocols require that zero tolerance defects remain below the threshold specified in the performance standard.",
        "ensure_stunning_vs_optimize_throughput", "line_speed"),
]


def generate_phase3_dataset(output_dir: str = None) -> dict:
    """Generate the complete Phase 3 dataset."""
    sub_types = {
        "practical_compassion": PRACTICAL_PAIRS,
        "emotional_manipulation": EMOTIONAL_PAIRS,
        "cross_domain": CROSS_DOMAIN_PAIRS,
        "acknowledged_dismissal": ACKNOWLEDGED_DISMISSAL_PAIRS,
        "sentimentality_vs_ethics": SENTIMENTALITY_VS_ETHICS_PAIRS,
        "understated_compassion": UNDERSTATED_COMPASSION_PAIRS,
    }

    total = 0
    for name, pairs in sub_types.items():
        print(f"  {name}: {len(pairs)} pairs")
        total += len(pairs)
    print(f"  Total Phase 3 pairs: {total}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        all_pairs_path = os.path.join(output_dir, "phase3_pairs.jsonl")
        with open(all_pairs_path, "w") as f:
            for st_name, pairs in sub_types.items():
                for pair in pairs:
                    f.write(json.dumps(asdict(pair)) + "\n")
        print(f"  Saved to: {all_pairs_path}")

    return sub_types


def get_all_phase3_pairs() -> List[TextPair]:
    """Return all Phase 3 pairs as a flat list."""
    return (PRACTICAL_PAIRS + EMOTIONAL_PAIRS + CROSS_DOMAIN_PAIRS +
            ACKNOWLEDGED_DISMISSAL_PAIRS + SENTIMENTALITY_VS_ETHICS_PAIRS +
            UNDERSTATED_COMPASSION_PAIRS)


if __name__ == "__main__":
    generate_phase3_dataset(
        output_dir=os.path.join(os.path.dirname(__file__), "datasets")
    )
