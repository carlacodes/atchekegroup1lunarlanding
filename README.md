# transferlanders' NMA Deep Learning Repository
<i> Transfer learning in the context of the lunar lander problem. </i>

## Welcome to the transferlearner's shared Lunar Lander RL repository.

## Sources cited:
<br> For creating a DQN model in Pytorch: Machine Learning with Phil
<br> Optimal hyperparameters for the DQN stablebaselines3 model: huggingface
<br> Copying parameters over from stablebaslines3 to pytorch: https://colab.research.google.com/drive/1XwCWeZPnogjz7SLW2kLFXEJGmynQPI-4?usp=sharing

Requirements are listed in package-list.txt

### abstract:
Since deep reinforcement learning models are highly optimized for a specific learning environment, they require extensive training and their performance decay when presented with new environments. 
Recently, Gadgil et al. [1] found that adding additional uncertainty to the OpenAI Gym LunarLander environment still produced positive rewards when agents were trained under a Deep Q Learning (DQN) model, suggesting that adversarial training may improve overall agent performance. 
We hypothesize that training agents on more complex, noisier environments will improve performance stability relative to the baseline agent trained in the default environment when they are placed in a novel, modified environment with obstacles. We used three tasks as our training environment based on the LunarLander task, a 2-D environment in which a lander aims to maximize reward through landing safely on a defined landing pad. 
Besides the original task, we modified this task by adding: 1) noise to the observations, and 2) a static obstacle. We trained a Deep Q Learning (DQN) agent separately on these three tasks to generate a pre-trained model. 
We also completed hyperparameter tuning and reward shaping to optimize the performance of the pre-trained model. <br> We then used the model pre-trained using noisy observations and the model pre-trained on the default environment in the obstacle task. The overall goal was to see whether what is learned from one task can be adapted to another, and which environment has the more adaptive pre-trained agent. Future work will incorporate testing the performance of transfer learning with or without retraining or reinitializing layers using the transfer ratio metric. <br>
	[1]
S. Gadgil, Y. Xin, and C. Xu, “Solving The Lunar Lander Problem under Uncertainty using Reinforcement Learning.” arXiv, Nov. 23, 2020. Accessed: Jul. 26, 2022. [Online]. Available: http://arxiv.org/abs/2011.11850