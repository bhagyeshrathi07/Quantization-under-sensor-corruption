# Disagreements for ronak (88 labelled item(s) of 88 assigned)

Disagreements: 10 of 88. Agreement with ronak: fable 81/88, astra 82/88.
Human label against the model labels (fable, astra). The humans decide the consensus in discussion; a model label is a second opinion, never a consensus by itself.

## D003 (planning/VQA): ronak 1, fable 0, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_BACK,0.5511,0.5314>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed because there is no safety issue. The probability is high.

- fable (0, flagged ambiguous): The answer settles on staying in the current lane with a high probability of needing to brake rather than committing to keeping the same speed, so the reference action is not the recommended one.
- astra (0): Staying in the current lane does not supply the required recommendation to keep going at the same speed.

**Consensus:** _____ (reason: _____)

## D019 (planning/VQA): ronak 1, fable 0, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_FRONT,0.6739,0.5750>? Why take this action and what's the probability?

**Reference:** The action is to remain stationary. The reason for this action is to follow traffic rules. The probability of this action is low.

- fable (0): The recommended action (stop) matches remaining stationary in meaning, but the stated probability is high while the reference says low, so the probability points the other way.
- astra (0): The answer assigns a high probability to stopping, whereas the reference assigns a low probability to remaining stationary.

**Consensus:** _____ (reason: _____)

## D032 (planning/VQA): ronak 1, fable 0, astra 0

**Question:** In this scenario, what are safe actions to take for the ego vehicle?

**Reference:** Brake gently to a stop, brake suddenly.

- fable (0, flagged ambiguous): The answer recommends coming to a stop, which covers braking gently to a stop, but the reference's second action, brake suddenly, is absent and the answer is cut off before completing.
- astra (0): The answer recommends stopping but does not specify both gentle braking to a stop and sudden braking.

**Consensus:** _____ (reason: _____)

## D036 (planning/VQA): ronak 1, fable 1, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_FRONT_RIGHT,0.7146,0.5602>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is that there is no safety issue, and the probability of taking this action is high.

- fable (1, flagged ambiguous): The answer settles on continuing straight as the most probable action, matching keep going at the same speed, though it then calls the probability moderate rather than high.
- astra (0): The answer recommends continuing straight without specifying maintaining the same speed and explicitly assigns moderate probability rather than high.

**Consensus:** _____ (reason: _____)

## D044 (planning/VQA): ronak 0, fable 1, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_BACK,0.5558,0.5324>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is that there is no safety issue. The probability is high.

- fable (1, flagged ambiguous): The answer settles on following the truck ahead at its steady pace with high probability, which reads as keeping the current speed, though it never states the speed explicitly.
- astra (0, flagged ambiguous): Although following a steadily moving truck could suggest constant speed, the answer does not clearly recommend maintaining the ego vehicle's current speed.

**Consensus:** _____ (reason: _____)

## D048 (planning/VQA): ronak 1, fable 1, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_FRONT,0.8292,0.5250>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is that there is no safety issue. The probability is high.

- fable (1): The answer commits to proceeding straight as the most likely action with high probability, which matches keep going at the same speed, and the stop option is only a fallback it does not settle on.
- astra (0): The answer assigns high likelihood to proceeding straight but does not specify maintaining the same speed.

**Consensus:** _____ (reason: _____)

## D053 (planning/VQA): ronak 1, fable 1, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_BACK,0.5104,0.6028>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is to avoid a collision, which is high in probability.

- fable (1): The answer commits to continuing straight ahead in the current lane and states a high probability for that action, matching the reference.
- astra (0): The answer predicts continuing straight but does not state that the ego vehicle will maintain the same speed.

**Consensus:** _____ (reason: _____)

## D066 (planning/VQA): ronak 0, fable 1, astra 0

**Question:** What actions taken by the ego vehicle can lead to a collision with <c1,CAM_FRONT,0.4161,0.5547>?

**Reference:** Accelerating and going straight.

- fable (1, flagged ambiguous): The answer says moving forward into the street without stopping leads to a collision with the white car, which I read as accelerating and going straight from a parked position, though 'accelerating' is not stated outright.
- astra (0): The answer identifies moving forward as dangerous but does not identify accelerating as a collision-causing action.

**Consensus:** _____ (reason: _____)

## D074 (planning/VQA): ronak 0, fable 1, astra 0

**Question:** In this scenario, what are safe actions to take for the ego vehicle?

**Reference:** Keep going at the same speed, and brake gently to a stop.

- fable (1, flagged ambiguous): The answer recommends maintaining speed and slowing down ready to stop for the pedestrian, which covers both reference actions in meaning, though 'brake gently to a stop' is only conditional in the answer.
- astra (0, flagged ambiguous): The answer recommends slowing down and preparing to stop but does not clearly recommend braking gently to a stop.

**Consensus:** _____ (reason: _____)

## D082 (planning/VQA): ronak 0, fable 1, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_BACK,0.5104,0.6028>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is to avoid a collision, which is high in probability.

- fable (1, flagged ambiguous): The answer settles on continuing straight as the most likely action with high probability, which reads as keeping going at the same speed, and its reason does not contradict the reference.
- astra (0): The answer assigns high likelihood to continuing straight but does not state that the vehicle will maintain the same speed.

**Consensus:** _____ (reason: _____)

