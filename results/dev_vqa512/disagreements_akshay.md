# Disagreements for akshay (87 labelled item(s) of 87 assigned)

Disagreements: 14 of 87. Agreement with akshay: fable 78/87, astra 75/87.
Human label against the model labels (fable, astra). The humans decide the consensus in discussion; a model label is a second opinion, never a consensus by itself.

## D179 (planning/VQA): akshay 1, fable 0, astra 0

**Question:** What actions could the ego vehicle take based on <c2,CAM_FRONT_LEFT,0.5418,0.5173>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is that the object has little reference value for the ego vehicle, with a high probability.

- fable (0): The answer settles on stopping and waiting, the opposite of the reference's keep going at the same speed.
- astra (0): The answer recommends stopping and waiting rather than keeping the same speed.

**Consensus:** _____ (reason: _____)

## D183 (planning/VQA): akshay 0, fable 1, astra 0

**Question:** What actions taken by the ego vehicle can lead to a collision with <c1,CAM_FRONT,0.4161,0.5547>?

**Reference:** Accelerating and going straight.

- fable (1, flagged ambiguous): The answer names accelerating toward the car ahead as a collision-causing action, which covers accelerating and going straight, though it never says 'straight' explicitly.
- astra (0, flagged ambiguous): Acceleration is identified as dangerous, but going straight is only arguably implied by the stopping-distance discussion rather than identified as a collision-causing action.

**Consensus:** _____ (reason: _____)

## D187 (perception/VQA): akshay 1, fable 0, astra 0

**Question:** What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.

**Reference:** There is a yellow sedan to the back of the ego vehicle, a person riding a motorcycle to the back right of the ego vehicle, a yellow bus to the front right of the ego vehicle, and a yellow roadblock to the front of the ego vehicle. The IDs of these objects are <c1,CAM_BACK,0.6323,0.6314>, <c2,CAM_BACK_RIGHT,0.5146,0.7500>, <c3,CAM_FRONT_RIGHT,0.7364,0.4342>, and <c4,CAM_FRONT,0.9389,0.3974>.

- fable (0): The yellow sedan behind and the yellow roadblock in front are missing; only the motorcycle and bus appear.
- astra (0): The answer omits the yellow sedan behind and the yellow roadblock ahead.

**Consensus:** _____ (reason: _____)

## D190 (planning/VQA): akshay 1, fable 1, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_FRONT_RIGHT,0.7146,0.5602>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is that there is no safety issue, and the probability of taking this action is high.

- fable (1, flagged ambiguous): The answer settles on continuing straight with high probability, which matches keeping going at the same speed and a high likelihood, though it does not mention speed explicitly.
- astra (0): The answer assigns high likelihood to continuing straight but does not specify maintaining the same speed.

**Consensus:** _____ (reason: _____)

## D192 (planning/VQA): akshay 1, fable 0, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_BACK,0.6323,0.6314>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is that there is no safety issue, and the probability is high.

- fable (0): The answer lists several conditional actions without committing to one and says the probability cannot be determined.
- astra (0): The answer lists conditional alternatives without selecting maintaining the same speed as the highly likely action.

**Consensus:** _____ (reason: _____)

## D195 (planning/VQA): akshay 1, fable 1, astra 0

**Question:** What actions taken by the ego vehicle can lead to a collision with <c3,CAM_FRONT,0.4748,0.5524>?

**Reference:** Accelerating and going straight can lead to a collision with <c3,CAM_FRONT,0.4748,0.5524>.

- fable (1, flagged ambiguous): The answer concludes that accelerating (or changing lanes) can cause a collision with the van ahead in the same lane, which covers accelerating and going straight.
- astra (0, flagged ambiguous): The answer identifies acceleration as collision-causing, but its description of current forward movement only arguably supplies the required going-straight action.

**Consensus:** _____ (reason: _____)

## D203 (planning/VQA): akshay 0, fable 1, astra 0

**Question:** What actions taken by the ego vehicle can lead to a collision with <c1,CAM_FRONT,0.5120,0.6000>?

**Reference:** Accelerating and going straight.

- fable (1, flagged ambiguous): Answer names speeding (accelerating) toward the white car ahead in the same direction as a collision-causing action, which reads as accelerating and going straight.
- astra (0, flagged ambiguous): The answer identifies acceleration as a collision risk but does not clearly identify going straight as part of the collision-causing action.

**Consensus:** _____ (reason: _____)

## D206 (planning/VQA): akshay 1, fable 0, astra 0

**Question:** What actions taken by the ego vehicle can lead to a collision with <c1,CAM_FRONT_RIGHT,0.7146,0.5602>?

**Reference:** None.

- fable (0, flagged ambiguous): Reference is None, but the answer lists turning right, slowing and accelerating as actions that could lead to a collision; the cut-off final sentence does not reach a no-collision conclusion.
- astra (0): The answer explicitly names applicable collision-causing actions and never concludes that no such action is indicated.

**Consensus:** _____ (reason: _____)

## D227 (planning/VQA): akshay 1, fable 1, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_FRONT,0.6604,0.6314>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is to avoid a collision, which is high in probability.

- fable (1, flagged ambiguous): The answer settles on continuing straight as the most likely action (high, 80%), which reads as the reference's keep going with a likelihood pointing the same way, though it never says 'at the same speed'.
- astra (0): The answer assigns high likelihood to continuing straight but does not state that the ego vehicle will maintain its current speed.

**Consensus:** _____ (reason: _____)

## D230 (planning/VQA): akshay 1, fable 0, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_BACK,0.5073,0.5778>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is that there is no safety issue, and the probability is high.

- fable (0, flagged ambiguous): The answer leaves it open between maintaining speed and accelerating without settling on one action, even though it rates maintaining speed as high probability.
- astra (0): The conclusion offers maintaining speed or accelerating as alternatives without settling on the reference action.

**Consensus:** _____ (reason: _____)

## D237 (planning/VQA): akshay 1, fable 0, astra 0

**Question:** What actions taken by the ego vehicle can lead to a collision with <c1,CAM_BACK_LEFT,0.2156,0.6963>?

**Reference:** No action taken by the ego vehicle will lead to a collision with <c1,CAM_BACK_LEFT,0.2156,0.6963>.

- fable (0): Against a None-type reference the answer names an applicable collision-causing ego action (not maintaining a safe following distance) and gives no conclusion that none applies.
- astra (0): The answer identifies applicable collision risks instead of explicitly concluding that no collision-causing action is indicated.

**Consensus:** _____ (reason: _____)

## D243 (planning/VQA): akshay 1, fable 0, astra 0

**Question:** In this scenario, what are safe actions to take for the ego vehicle?

**Reference:** Keep going at the same speed, decelerate gradually without braking.

- fable (0): The answer recommends slowing down and braking if needed but never keeping the same speed, so a reference action is missing and braking contradicts decelerating without braking.
- astra (0): The answer omits maintaining the same speed and does not specify gradual deceleration without braking.

**Consensus:** _____ (reason: _____)

## D246 (planning/VQA): akshay 1, fable 1, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_BACK,0.5130,0.6352>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is to avoid a collision, which is high probability.

- fable (1, flagged ambiguous): The answer settles on continuing straight as the most probable action with high probability, which I read as keeping the current speed since it lists accelerate and stop as separate low-probability options.
- astra (0): The answer selects continuing straight with high likelihood but does not specify maintaining the same speed.

**Consensus:** _____ (reason: _____)

## D249 (planning/VQA): akshay 1, fable 1, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_FRONT,0.0469,0.6453>? Why take this action and what's the probability?

**Reference:** The action is to decelerate gradually without braking. The reason is to avoid a collision. The probability is high.

- fable (1, flagged ambiguous): The answer commits to slowing down (decelerating) with a stated high probability, and its yield reason does not contradict avoiding a collision; the added stop-if-necessary is conditional.
- astra (0, flagged ambiguous): The selected action is slowing down and possibly stopping, which does not clearly establish gradual deceleration without braking.

**Consensus:** _____ (reason: _____)

