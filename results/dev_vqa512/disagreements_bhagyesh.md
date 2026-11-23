# Disagreements for bhagyesh (87 labelled item(s) of 87 assigned)

Disagreements: 6 of 87. Agreement with bhagyesh: fable 84/87, astra 81/87.
Human label against the model labels (fable, astra). The humans decide the consensus in discussion; a model label is a second opinion, never a consensus by itself.

## D091 (planning/VQA): bhagyesh 0, fable 1, astra 1

**Question:** What actions taken by the ego vehicle can lead to a collision with <c1,CAM_BACK,0.5104,0.6028>?

**Reference:** Brake suddenly.

- fable (1, flagged ambiguous): The answer names sudden braking ('brakes suddenly without sufficient deceleration ... may collide') as a collision-causing action, though it frames it as braking too late against a vehicle ahead rather than the reference's bare 'brake suddenly'.
- astra (1): The answer explicitly includes braking suddenly as an action that may lead to a collision.

**Consensus:** _____ (reason: _____)

## D096 (planning/VQA): bhagyesh 1, fable 1, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_BACK_RIGHT,0.3547,0.7130>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is that the object has little reference value for the ego vehicle, high probability.

- fable (1): The answer settles on continuing straight as the most probable action with high probability, matching 'keep going at the same speed, high probability', and its reason does not contradict the reference.
- astra (0): The answer assigns high likelihood to continuing straight but does not specify maintaining the same speed.

**Consensus:** _____ (reason: _____)

## D099 (planning/VQA): bhagyesh 0, fable 1, astra 1

**Question:** What actions taken by the ego vehicle can lead to a collision with <c2,CAM_FRONT_RIGHT,0.3442,0.5444>?

**Reference:** Accelerating and going straight.

- fable (1, flagged ambiguous): 'Sudden acceleration while still following the truck' matches 'accelerating and going straight' as a collision-causing action; the closing hedge does not deny risk from that action.
- astra (1, flagged ambiguous): Rapid acceleration while following the truck plausibly conveys accelerating and going straight as a collision-causing action, although straight travel is implicit.

**Consensus:** _____ (reason: _____)

## D127 (planning/VQA): bhagyesh 1, fable 0, astra 0

**Question:** What actions taken by the ego vehicle can lead to a collision with <c1,CAM_FRONT,0.5120,0.6000>?

**Reference:** Accelerating and going straight.

- fable (0): The answer names failure to detect the car ahead and not braking or changing lanes; accelerating, a reference action, is missing.
- astra (0): The answer discusses high speed and failure to brake but does not identify accelerating as a collision-causing action.

**Consensus:** _____ (reason: _____)

## D149 (planning/VQA): bhagyesh 1, fable 1, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_BACK,0.5130,0.6352>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is to avoid a collision, which is high probability.

- fable (1): The answer settles on continuing straight (keep going) as the most likely action with an explicitly high probability, and its reason does not contradict the reference.
- astra (0): Although it assigns high probability to continuing straight, the answer does not state that the ego vehicle will maintain its current speed.

**Consensus:** _____ (reason: _____)

## D175 (planning/VQA): bhagyesh 1, fable 1, astra 0

**Question:** What actions could the ego vehicle take based on <c1,CAM_FRONT_LEFT,0.6370,0.5797>? Why take this action and what's the probability?

**Reference:** The action is to keep going at the same speed. The reason is that there is no safety issue, and the probability is high.

- fable (1, flagged ambiguous): The answer settles on continuing straight as the most likely action with high probability, which matches keeping going at the same speed in meaning.
- astra (0): The answer gives a high likelihood of continuing straight but does not recommend maintaining the same speed.

**Consensus:** _____ (reason: _____)

