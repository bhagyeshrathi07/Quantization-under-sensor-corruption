# Labelling guide for the judge calibration sheet

Rules fixed on November 17, 2026 (drafted November 16, the three open choices settled on the reviewer's recommendation; the choices were made before any label existed and not to improve the acceptance rate). This is the study's own, independent correctness criterion for a model answer, the "rubric's correctness criterion" of decision 3 in `judge_decisions_2026-11-13.md`. It is not the judge's scoring scheme: the released judge templates score other things as well (the planning template awards points for the probability of the action being safe or feasible, the perception template awards action and motion points), and they stay exactly as released during the calibration. The calibration gate of decision 4 tests whether the judge's scores predict this criterion well enough; if they do not, VQA stays unassessed. `scripts/13_label_ui.py` shows the cell's section beside every item. The sheets of November 14 are withdrawn (the 512-token amendment, `explore_512tok_2026-11-16.md`); this guide applies to `judge_label_sheet_2026-11-17.csv`. Nothing here changes once labelling has started; a change afterwards would mean labels given under two criteria.

Agreed by: Ronak Patel (November 17, 2026); Bhagyesh Rathi (November 20, 2026); Akshay: pending, to be recorded here before they label.

Who labels what (November 19): one person per item, in blocks (`judge_label_sheet_2026-11-17.assignment.json`): Ronak D001 to D088, Bhagyesh D089 to D175, Akshay D176 to D262. The tool shows you your block only. Label alone, in order; the model labellers' opinions are shown only after your whole block is committed, and the three of you settle the disagreements together.

## What the label means

One label per item: `1` the model's answer is correct, `0` it is not. Correct means: read against the reference answer (the dataset's ground truth), the model's answer states the same core content. The reference is the standard, not the scene: no labeller sees the images, exactly as the judge does not. Where the reference itself looks wrong, label against the reference anyway and note the item for the discussion.

Only the content decides. Length, style, markdown, headings, confidence words and spelling never decide. The model's answers are long and the references are short: find the reference's content inside the answer. A long answer that contains the reference's core content and adds other material is correct unless the additions contradict the reference. The judge's scores are not visible and must not be guessed at.

## Common rules

- Paraphrase: the same meaning in other words is a match ("slow down" and "decelerate", "go straight" and "continue ahead", "keep going at the same speed" and "maintain the current speed", "the car in front" and "the vehicle ahead").
- Partial answer: every core element of the reference must be present. One core element missing or wrong gives `0`. Extra correct detail does not give or take anything.
- Missing detail that is not core (a reason when only the action was asked, a distance, a colour the reference does not give): ignore.
- Unsupported claims: a claim the reference neither contains nor contradicts is ignored (we cannot check it without the scene). A claim that contradicts the reference gives `0`.
- Object references in the question such as `<c1,CAM_BACK,0.53,0.50>` name an object by camera and image position. The model may not identify it; label the answer's action or object against the reference regardless. The model is never required to reproduce these IDs.
- Hedging: an answer that commits to the reference's content with softeners ("probably", "should", "may need to") is a match. An answer that offers alternatives ("turn left or go straight") is `0` unless the reference offers the same alternatives.
- Contradictions inside the answer (says "stop" then "keep moving"): `0`.
- Empty answer, refusal, "I cannot determine this from the image", a question back, an answer to a different question: `0`.
- Cut-off answers: some answers end at the generation cap, mid-sentence. The judge sees the same text. Apply the agreed rubric to the text shown: a core element the answer did not reach is missing, so `0`; do not infer how an unfinished answer would continue.
- Numbers and counts: when the reference's point is a number, it must match. Approximate wording ("a few") against an exact reference gives `0`.
- When torn after one careful read, pick the label you lean to and keep going; the discussion covers disagreements. Do not skip an item: the tool does not allow it, and a blank breaks the sheet-order rule.

## perception/VQA

All 60 items ask "What are the important objects in the current scene?". The reference lists the objects with a short description and a position relative to the ego vehicle ("a white van to the front of the ego vehicle, a black car to the back ..."), then their IDs. Core elements: the objects the reference names, each recognizable in the answer by category and, where the reference gives them, colour and position.

- `1`: every object the reference names appears in the answer with a matching category (a "white van" for "a white van"; "a cyclist" for "a person riding a bicycle"; a generic "vehicle" matches a car, sedan, SUV, van or truck only when colour or position confirms it), and no stated position or colour contradicts the reference.
- `0`: a reference object missing from the answer, or named with a contradicting colour or position (reference "black car to the back", answer "black car ahead").
- Objects the answer adds (buildings, lane markings, signs, far traffic) are ignored. The IDs are never required.
- Rule chosen: every reference object is required (an answer that lists four of five is `0`), matching the completeness rule above. Clear paraphrases count; each reference object is matched separately to one distinct object in the answer (one "a vehicle" in the answer does not cover two reference vehicles).

## prediction/VQA

The 22 items ask which object to notice first, second and third, each one's state and the action to take. The reference has the form "Firstly, notice that white truck at left camera. The object is stationary, so the ego vehicle should continue ahead at the same speed. Secondly, ... Thirdly, ...". Core elements: the three objects, their order, each object's state, each action.

- `1`: the same three objects in the same order (category-level match counts: "pedestrians" for "pedestrians crossing", "a sedan" for "silver sedan"; the camera side is a position hint, not a required phrase), each with a state matching in meaning (stationary versus moving; "going ahead" is moving) and an action matching in meaning ("continue at the same speed" and "keep going" match; "slow down" does not match "continue ahead at the same speed").
- `0`: a missing or swapped object, a wrong state for any object, or an action that contradicts the reference for any object.
- The answers are long and often structured by image rather than by object; check object by object against the reference and ignore everything else.

## planning/VQA

Three question types among the 180 items.

"In this scenario, what are safe actions to take for the ego vehicle?" (60 items). The reference is a short list of actions ("Keep going at the same speed and continue ahead"; "Decelerate gradually without braking"; "Brake gently to a stop, brake suddenly"). Core elements: the reference's actions.

- `1`: every action the reference names is among the actions the answer recommends as safe (in meaning), and the answer does not recommend as safe an action that contradicts one of them (reference "keep going at the same speed", answer "reduce speed" as the recommendation: contradiction).
- `0`: a reference action missing from the recommendations, or a contradicting recommendation. General advice ("be aware of surroundings", "maintain a safe distance") is neither an action of the reference nor a contradiction; ignore it.

"What actions could the ego vehicle take based on <object>? Why take this action and what's the probability?" (60 items). The reference has the form "The action is to keep going at the same speed. The reason is that there is no safety issue. The probability is high." Core elements: the action, and the probability's direction (high or low).

- `1`: the answer's recommended action matches the reference's in meaning, and the answer states how likely the ego vehicle is to take that action, pointing the same way as the reference (high, likely, most probable versus low, unlikely). The reason must not contradict the reference's reason; a differently worded reason is fine.
- `0`: a different or opposite action, a probability pointing the other way, or no action at all. An answer that lists several possible actions without recommending one is `0` unless the reference's action is the one it settles on.
- Rule chosen: the probability element stays and means the likelihood that the ego vehicle takes that action (the dataset's meaning, DriveLM planning), not that the action is safe. The answer must state that likelihood for the chosen action ("most likely action", "will probably continue", "high probability of keeping its speed"), pointing the same way as the reference. Wording about safety alone ("this is safe", "low risk") does not supply it, and a bare "high confidence" counts only when it is explicitly about taking that action. An answer with the right action and no likelihood statement is `0`.

"What actions taken by the ego vehicle can lead to a collision with <object>?" (60 items). The reference names the dangerous actions ("Back up"; "Accelerating and going straight"). Core elements: those actions.

- `1`: every action the reference names appears in the answer as a collision-causing action (in meaning: "reversing" for "back up"; "speeding up while continuing forward" for "accelerating and going straight").
- `0`: a reference action missing, or an answer that concludes there is no collision risk from the reference's actions.
- References that say "None." (rule chosen): the dataset says no action of the ego vehicle in the described scenario leads to a collision with that object. `1` requires an explicit conclusion that no collision-causing action is indicated for that object in this scenario (or that the object poses no collision risk). Naming an applicable collision-causing action, or giving no conclusion (listing hypothetical dangers and stopping), is `0`. The answer need not claim that a collision is impossible under every imaginable action.

## Practice items (synthetic, not from the sheet)

Read these together before labelling. Reference first, then the answer, then the label under the rules above (reviewer-checked November 17: 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0).

1. perception. Reference: "There is a white van to the front of the ego vehicle and a black sedan to the back of the ego vehicle." Answer lists "a white van ahead, parked under a canopy; a black car visible behind; a multi-storey building; lane markings". Label `1` (both objects present, extras ignored).
2. perception. Same reference. Answer lists "a white van ahead; a bus approaching; buildings". Label `0` (black sedan missing).
3. perception. Reference: "a black car to the front of the ego vehicle". Answer: "a black car directly behind the ego vehicle". Label `0` (position contradicts).
4. prediction. Reference: "Firstly, notice that pedestrian at front camera. The object is stationary, so the ego vehicle should stop. Secondly, notice that white car at left camera. The object is going ahead, so the ego vehicle should keep going at the same speed. Thirdly, notice that traffic light at front camera. The object is red, so the ego vehicle should wait." Answer: "First the pedestrian ahead, standing still: stop. Then the white car on the left, moving: maintain speed. Then the red light: wait." Label `1`.
5. prediction. Same reference. Answer: "First the white car on the left, moving, keep going; second the pedestrian, stationary, stop; third the red light, wait." Label `0` (order swapped).
6. planning, safe actions. Reference: "Keep going at the same speed and continue ahead." Answer recommends "maintain the current speed and lane, keep a safe distance, monitor the surroundings". Label `1` (reference action present; the rest is general advice).
7. planning, safe actions. Same reference. Answer recommends "reduce speed because the road is wet, maintain distance". Label `0` (recommends the contradicting action, the reference's action absent).
8. planning, based on object. Reference: "The action is to turn left because there is no safety issue. The probability of this action is high." Answer: "The ego vehicle will most likely turn left; the lane is clear." Label `1`. The same answer worded "The ego vehicle can turn left; this is safe" is `0` under the proposed rule: safety wording is not a likelihood of the action.
9. planning, based on object. Same reference. Answer: "The ego vehicle could continue straight, turn left or stop, depending on the signal." Label `0` (no commitment to the reference's action).
10. planning, collision. Reference: "Back up." Answer: "Reversing without checking the rear camera would hit the vehicle behind." Label `1` ("reversing" is "back up").
11. planning, collision. Reference: "Accelerating and going straight." Answer: "There is no collision risk with this object as long as the ego vehicle keeps its lane." Label `0` (concludes no risk).
12. any cell. Answer: "I cannot determine this from the image." Label `0`.
13. planning, collision, reference "None.". Answer: "The parked car on the right is well off the ego vehicle's path; no action the ego vehicle would take here leads to a collision with it." Label `1` (explicit conclusion of no collision-causing action in this scenario).
14. planning, collision, reference "None.". Answer: "A collision could occur if the ego vehicle drifts right, accelerates suddenly, or fails to brake in time." Label `0` (names applicable collision-causing actions; no conclusion that none applies).

## After labelling

Each person's labels stay in their own copy until `scripts/12_merge_label_copies.py` merges them. `scripts/11_calibrate_judge.py --labels-only` gives the agreement before the discussion. The discussion goes item by item through the disagreements among the items all three reached, records the consensus and a one-line reason where somebody changed their mind, and leaves the pre-discussion labels as they were.
