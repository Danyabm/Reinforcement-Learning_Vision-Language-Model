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

**Results**
**Problem 1.1: Training & Evaluation**
Success Rate: 27.0% (27/100 episodes)

First successful episode: 32
Training: PPO with 64 parallel environments
Total timesteps: 500,000

Success Count: 27/100 episodes
Success Rate: 27.0%
Video: success.mp4 (Episode 32)
Problem 1.2: VLM Task Description
Initial Scene Description:

"The image depicts a robotic arm, specifically a robotic arm of a manufacturing robot, performing a task on a wooden table, where it is carefully placing a small cube on a target, which is marked with a bullseye symbol, indicating that the task is to accurately place the cube on the target, as evidenced by the robotic arm's precise movements and the cube's position on the target."

**Problem 2: VLM Goal Detection**
Final Scene Analysis:

"Based on the image, I can see that the robot is attempting to pull the blue cube towards the target placed on the table. However, the cube is not yet aligned with the target, and it appears that the robot is still in the process of moving the cube. To determine if the task is completed, I would need to see if the blue cube is now aligned with the target and is in close proximity to it. If the cube is still not aligned with the target, then the task is not yet completed."

*Model Used**: Llama 3.1-8B Vision (qresearch/llama-3.1-8B-vision-378)
