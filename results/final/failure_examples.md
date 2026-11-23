# Failure examples: harmful MCQ flips (BF16 correct, INT4 incorrect), 12 of 55

Source: `results/final/compare.json` and the raw answers in `results/final`. Lowest BF16 margin first, at most 8 per cell and 5 per condition. The margin is BF16's probability gap between its top two options at the first token; the pre-registered contrast C1 found flips concentrated in the lowest margin bin.

## 1. perception/MCQ, clean, margin 0.000

Question `1fbcc26e-03d203bc-e4cdeb1b70`, scene `1fbcc26e`, 1 view(s).

**What is the moving status of object <c1,CAM_FRONT,0.3042,0.5408>? Please select the correct answer from the following**

- A. Going ahead. (correct)
- B. Turn left.
- C. Turn right.

BF16 answered **A** (option probabilities at the first token: A 0.34, B 0.34, C 0.27); INT4 answered **C**.

BF16 text: A  The object at coordinates <c1,CAM_FRONT,0.3042,0.5408> appears to be a vehicle in the distance, traveling along the road. Given its position relative to the road markings and the direction it seems to be heading, it is likely going ahead. There is no indication that it is turning left or right. T

INT4 text: C  The object located at coordinates <c1,CAM_FRONT,0.3042,0.5408> appears to be a vehicle positioned near the center of the road. Given its position relative to the lane markings and the absence of any visible indicators suggesting immediate turning intentions, it seems to be moving straight ahead. 

Views: CAM_FRONT (missing)

## 2. behavior/MCQ, Fog, margin 0.000

Question `240d1548-e154fa0b-1bc4fccc56`, scene `240d1548`, 6 view(s).

**Predict the behavior of the ego vehicle. Please select the correct answer from the following**

- A. The ego vehicle is steering to the right. The ego vehicle is driving slowly.
- B. The ego vehicle is slightly steering to the right. The ego vehicle is driving slowly. (correct)
- C. The ego vehicle is steering to the left. The ego vehicle is driving with normal speed.
- D. The ego vehicle is steering to the right. The ego vehicle is driving very fast.

BF16 answered **B** (option probabilities at the first token: A 0.26, B 0.29, C 0.29, D 0.16); INT4 answered **A**.

BF16 text: B

INT4 text: A

Views: CAM_FRONT (missing), CAM_FRONT_LEFT (missing), CAM_FRONT_RIGHT (missing), CAM_BACK (missing), CAM_BACK_LEFT (missing), CAM_BACK_RIGHT (missing)

## 3. behavior/MCQ, Fog, margin 0.000

Question `4ed62829-c16bf051-fb93e76780`, scene `4ed62829`, 6 view(s).

**Predict the behavior of the ego vehicle. Please select the correct answer from the following**

- A. The ego vehicle is going straight. The ego vehicle is driving fast. (correct)
- B. The ego vehicle is steering to the right. The ego vehicle is driving very fast.
- C. The ego vehicle is steering to the left. The ego vehicle is driving slowly.
- D. The ego vehicle is going straight. The ego vehicle is driving very fast.

BF16 answered **A** (option probabilities at the first token: A 0.36, B 0.03, C 0.36, D 0.24); INT4 answered **C**.

BF16 text: A

INT4 text: C

Views: CAM_FRONT (missing), CAM_FRONT_LEFT (missing), CAM_FRONT_RIGHT (missing), CAM_BACK (missing), CAM_BACK_LEFT (missing), CAM_BACK_RIGHT (missing)

## 4. perception/MCQ, clean, margin 0.000

Question `57f5503b-27ec52e7-d84762f12f`, scene `57f5503b`, 1 view(s).

**What is the moving status of object <c2,CAM_FRONT,0.3609,0.5676>? Please select the correct answer from the following**

- A. Turn left.
- B. Going ahead. (correct)
- C. Turn right.

BF16 answered **B** (option probabilities at the first token: A 0.27, B 0.35, C 0.35); INT4 answered **C**.

BF16 text: B. Going ahead.  The object located at coordinates <c2,CAM_FRONT,0.3609,0.5676> appears to be a vehicle that is currently in the process of turning left. This can be inferred from the position of the vehicle relative to the road markings and the presence of a left-turn arrow on the road. The vehicle

INT4 text: C  The object <c2,CAM_FRONT,0.3609,0.5676> appears to be a vehicle that is currently turning right. This can be inferred from the direction of the arrow painted on the road and the position of the vehicle relative to the turn lane. The vehicle is positioned in the lane designated for turning right, 

Views: CAM_FRONT (missing)

## 5. behavior/MCQ, LowLight, margin 0.000

Question `73030fb6-8de7ec06-24e410c705`, scene `73030fb6`, 6 view(s).

**Predict the behavior of the ego vehicle. Please select the correct answer from the following**

- A. The ego vehicle is steering to the left. The ego vehicle is driving very fast.
- B. The ego vehicle is slightly steering to the right. The ego vehicle is driving very fast.
- C. The ego vehicle is steering to the left. The ego vehicle is driving with normal speed. (correct)
- D. The ego vehicle is steering to the right. The ego vehicle is driving slowly.

BF16 answered **C** (option probabilities at the first token: A 0.25, B 0.12, C 0.32, D 0.32); INT4 answered **D**.

BF16 text: C

INT4 text: D

Views: CAM_FRONT (missing), CAM_FRONT_LEFT (missing), CAM_FRONT_RIGHT (missing), CAM_BACK (missing), CAM_BACK_LEFT (missing), CAM_BACK_RIGHT (missing)

## 6. perception/MCQ, Fog, margin 0.000

Question `78c5f7d7-4c4e1e94-7c5b6c0951`, scene `78c5f7d7`, 1 view(s).

**What is the moving status of object <c2,CAM_BACK_LEFT,0.3286,0.5861>? Please select the correct answer from the following**

- A. Turn right.
- B. Going ahead. (correct)
- C. Turn left.

BF16 answered **B** (option probabilities at the first token: A 0.05, B 0.46, C 0.46); INT4 answered **C**.

BF16 text: B. Going ahead.  The object in the specified coordinates appears to be a car that is currently moving forward on the road. There are no visible signs of turning or stopping, such as lane changes or brake lights being activated. The car's position relative to the road markings suggests it is maintain

INT4 text: C  The object <c2,CAM_BACK_LEFT,0.3286,0.5861> appears to be a car that is positioned on the left side of the intersection, facing towards the right side of the image. Given its position and orientation, it seems to be preparing to turn left at the intersection. Therefore, the correct answer is C. T

Views: CAM_BACK_LEFT (missing)

## 7. behavior/MCQ, clean, margin 0.000

Question `79770033-fbc07f14-d659520519`, scene `79770033`, 6 view(s).

**Predict the behavior of the ego vehicle. Please select the correct answer from the following**

- A. The ego vehicle is steering to the right. The ego vehicle is driving very fast.
- B. The ego vehicle is going straight. The ego vehicle is driving slowly. (correct)
- C. The ego vehicle is going straight. The ego vehicle is driving very fast.
- D. The ego vehicle is slightly steering to the left. The ego vehicle is driving with normal speed.

BF16 answered **B** (option probabilities at the first token: A 0.07, B 0.42, C 0.08, D 0.42); INT4 answered **D**.

BF16 text: B. The ego vehicle is going straight. The ego vehicle is driving slowly.  The ego vehicle appears to be driving on a straight road as there are no visible signs of turning. The speed seems to be slow due to the darkness and lack of visible traffic, which suggests that the driver is being cautious.

INT4 text: D  The ego vehicle appears to be driving with normal speed as there are no signs of rapid acceleration or deceleration. Additionally, the vehicle seems to be maintaining a straight path without any indication of turning, suggesting that it is likely driving with normal speed and going straight.

Views: CAM_FRONT (missing), CAM_FRONT_LEFT (missing), CAM_FRONT_RIGHT (missing), CAM_BACK (missing), CAM_BACK_LEFT (missing), CAM_BACK_RIGHT (missing)

## 8. behavior/MCQ, NoImage, margin 0.000

Question `9ac6dff0-7f32be25-618b2e25be`, scene `9ac6dff0`, 6 view(s).

**Predict the behavior of the ego vehicle. Please select the correct answer from the following**

- A. The ego vehicle is slightly steering to the left. The ego vehicle is driving with normal speed. (correct)
- B. The ego vehicle is steering to the left. The ego vehicle is driving very fast.
- C. The ego vehicle is slightly steering to the right. The ego vehicle is driving slowly.
- D. The ego vehicle is slightly steering to the right. The ego vehicle is driving with normal speed.

BF16 answered **A** (option probabilities at the first token: A 0.44, B 0.03, C 0.06, D 0.44); INT4 answered **D**.

BF16 text: A  Explanation:  The front-left and front-right cameras show that the vehicle's left side appears to be slightly closer to the lane markings than the right side, indicating that the vehicle is slightly steering to the left. There are no visible signs of high speed such as blurred motion or oversteer

INT4 text: D  Explanation: Based on the images provided, there is no clear indication of significant steering movement towards either the left or right. The vehicle appears to be maintaining a straight path. Additionally, there are no signs of rapid acceleration or deceleration that would suggest the vehicle i

Views: none shown (NoImage control)

## 9. perception/MCQ, LowLight, margin 0.000

Question `d0b2d916-f1999ab8-59ac357ab5`, scene `d0b2d916`, 1 view(s).

**What is the moving status of object <c1,CAM_BACK_LEFT,0.7995,0.5463>? Please select the correct answer from the following**

- A. Going ahead. (correct)
- B. Turn left.
- C. Turn right.

BF16 answered **A** (option probabilities at the first token: A 0.32, B 0.28, C 0.32); INT4 answered **C**.

BF16 text: A  The object in the specified coordinates appears to be a street sign, specifically a "Stop" sign. Since there is no indication of motion or any other vehicle or object in the immediate vicinity that would suggest turning left or right, the most appropriate answer is that the object is "Going ahead

INT4 text: C  The image shows a street scene with no visible vehicles or pedestrians in motion. The road appears clear, and there are no indications of any vehicles turning left or right. Therefore, it is not possible to determine the moving status of any specific object based on this image alone. However, if 

Views: CAM_BACK_LEFT (missing)

## 10. perception/MCQ, clean, margin 0.000

Question `d90b94e8-786371ca-0a2bbe9b63`, scene `d90b94e8`, 1 view(s).

**What is the moving status of object <c1,CAM_BACK,0.8917,0.5787>? Please select the correct answer from the following**

- A. Going ahead.
- B. Turn right. (correct)
- C. Turn left.

BF16 answered **B** (option probabilities at the first token: A 0.27, B 0.35, C 0.35); INT4 answered **C**.

BF16 text: B. Turn right.  The object in the coordinates <c1,CAM_BACK,0.8917,0.5787> appears to be a car that is positioned near the intersection, facing towards the right side of the image. Given its position relative to the road markings and the direction it seems to be heading, it is likely to turn right. T

INT4 text: C  The object located at coordinates <c1,CAM_BACK,0.8917,0.5787> appears to be a vehicle positioned near the intersection. Given its position relative to the road markings and the direction it seems to be heading towards, it looks like it is preparing to turn left. Therefore, the correct answer is C

Views: CAM_BACK (missing)

## 11. behavior/MCQ, LowLight, margin 0.000

Question `e5a3df5f-474e5ee2-a6c3f000be`, scene `e5a3df5f`, 6 view(s).

**Predict the behavior of the ego vehicle. Please select the correct answer from the following**

- A. The ego vehicle is going straight. The ego vehicle is driving with normal speed. (correct)
- B. The ego vehicle is steering to the right. The ego vehicle is driving slowly.
- C. The ego vehicle is going straight. The ego vehicle is not moving.
- D. The ego vehicle is going straight. The ego vehicle is driving slowly.

BF16 answered **A** (option probabilities at the first token: A 0.45, B 0.05, C 0.04, D 0.45); INT4 answered **D**.

BF16 text: A. The ego vehicle is going straight. The ego vehicle is driving with normal speed.

INT4 text: D

Views: CAM_FRONT (missing), CAM_FRONT_LEFT (missing), CAM_FRONT_RIGHT (missing), CAM_BACK (missing), CAM_BACK_LEFT (missing), CAM_BACK_RIGHT (missing)

## 12. behavior/MCQ, Fog, margin 0.036

Question `ef5f9458-cddb14d9-509e2d8e72`, scene `ef5f9458`, 6 view(s).

**Predict the behavior of the ego vehicle. Please select the correct answer from the following**

- A. The ego vehicle is steering to the left. The ego vehicle is driving slowly.
- B. The ego vehicle is slightly steering to the left. The ego vehicle is driving slowly.
- C. The ego vehicle is slightly steering to the right. The ego vehicle is driving slowly.
- D. The ego vehicle is going straight. The ego vehicle is not moving. (correct)

BF16 answered **D** (option probabilities at the first token: A 0.17, B 0.27, C 0.24, D 0.31); INT4 answered **C**.

BF16 text: D. The ego vehicle is going straight. The ego vehicle is not moving.  Explanation: In the provided images, there is no indication that the ego vehicle is moving. The vehicles around it appear stationary, and the smoke obscures any motion cues. Therefore, it can be inferred that the ego vehicle is no

INT4 text: C

Views: CAM_FRONT (missing), CAM_FRONT_LEFT (missing), CAM_FRONT_RIGHT (missing), CAM_BACK (missing), CAM_BACK_LEFT (missing), CAM_BACK_RIGHT (missing)

Images missing locally: 41 (listed in the JSON; fetch from the host with rsync before rendering).
