# roomba-rl
## Introduction

This is a custom Reinforcement Learning environment for training a Roomba style robot to do point-to-point navigation, including with obstacle avoidance. 

It can also [upload a learned hugging face RL algorithm trained on the custom environment to the Hugging face repository](https://huggingface.co/culteejen/PPO-default-RoombaAToB)

## Learned pitfalls:
- Need to change is_atari to use unregistered environments
- Make sure the action and observation space are accurate
- Make sure `reset()` returns ONLY obs! 

## TODO

- Add preview from links in readme
- Make sure new utils works
- Performance breakdown of model submission
- Fix model preview issue (model output not used correctly)
- Somehow, the particles' movements aren't that random (i.e. change of direction is constant)
