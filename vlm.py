import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

eval_vec_env = gym.make("PullCube-v1", num_envs=16, render_mode="rgb_array", control_mode= 'pd_joint_delta_pos')
eval_vec_env = RecordEpisode(eval_vec_env, output_dir="Videos", save_video=True,
save_trajectory=False, max_steps_per_video=max_episode_steps)
#eval_vec_env = ManiSkillSB3VectorEnv(eval_vec_env)
obs = eval_vec_env.reset()[0]
for i in range(max_episode_steps):
    action, _states = model.predict(obs.cpu().numpy(), deterministic=True)
    obs, rewards, dones,_, info = eval_vec_env.step(action)
    image_tensor = eval_vec_env.render()[0]  # torch tensor image as 256 x 256 x 3

# Convert torch tensor to PIL Image
image = Image.fromarray(image_tensor.cpu().numpy().astype('uint8'))
image.save("pushcube.jpg")

# Load Llama model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "qresearch/llama-3.1-8B-vision-378",  # use a pre-trained LLM model
    trust_remote_code=True,
    torch_dtype=torch.float16  # default data type for the weights of the model,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("qresearch/llama-3.1-8B-vision-378", use_fast=True)

# Generate description
description = model.answer_question(
    image, 
    "The task is to push the cube onto the target placed on the table. Check the image to see if the task has been completed.", 
    tokenizer,
    do_sample=True, temperature=0.3
)
print("Image description:", description)