# Reinforcement-Learning_Vision-Language-Model

**Problem 1: Train an RL Agent**

Pick a push task from ManiSkill
Train with RL algorithm (e.g., PPO)
Evaluate success rate
Record successful execution video

**Problem 2: VLM as Goal Detector**

Use VLM to describe the task
Check if task is completed

**Results & Demo**



https://github.com/user-attachments/assets/72627312-2791-4898-9719-b40d4495bfef



**Problem 1.1: Training & Evaluation**
Success Rate: 27.0% (27/100 episodes)

First successful episode: 32
Training: PPO with 64 parallel environments
Total timesteps: 500,000

Success Count: 27/100 episodes
Success Rate: 27.0%
Video: success.mp4 (Episode 32)

**Problem 1.2: VLM Task Description**

Initial Scene Description:

<img width="515" height="101" alt="image" src="https://github.com/user-attachments/assets/1ffbf6e1-41da-46d2-aa09-db28151a777a" />

<img width="221" height="221" alt="image" src="https://github.com/user-attachments/assets/56437d23-c628-4feb-b468-d3072e6c2b3f" />

**Problem 2: VLM Goal Detection**
Final Scene Analysis:

<img width="508" height="110" alt="image" src="https://github.com/user-attachments/assets/34428871-4998-4413-8dc4-5f6593a9d36a" />


"Based on the image, I can see that the robot is attempting to pull the blue cube towards the target placed on the table. However, the cube is not yet aligned with the target, and it appears that the robot is still in the process of moving the cube. To determine if the task is completed, I would need to see if the blue cube is now aligned with the target and is in close proximity to it. If the cube is still not aligned with the target, then the task is not yet completed."

*Model Used**: Llama 3.1-8B Vision (qresearch/llama-3.1-8B-vision-378)
