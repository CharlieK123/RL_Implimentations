# How to use!


90% of all meaningful code sits within: 
- PPO_model / PPO_model_revised (revised is specifically only for MuJoCo Humanoid and HumanoidStandup due to their odd action dim requirements)
- env_continuous / env_parallel (depends based on whether your chosen environment requires a discrete or continuous action space)

the other 10% consists of the feedforward neural net for critic and actor (feedforward2.py) and the reader.py file which is a bunch of GPT slop code, responcible for taking data from my PPO class, building the graph with all necessary metrics, and allowing to watch the current model act in the env.

TO RUN A TRAINING LOOP
1. run either env_continuous or env_paralell based on the desired env
2. set your hyperparameters with the PPO class
3. ensure you get the ID for the env on the gymnasium docs (really easy)
4. run the env and it will begin training and print a bunch of stuff.
5. while training run reader once you have set it up as requsted (same network build as the env file, same env id) otherwise it will give you errors
6. episode Exponential Moving Average is the most informative on performance, other metrics are there because they are necessary for debugging and training health


NOTE: hyperparameters are EXTREMELY sensitive to environments in classic RL fashion, a set of hyperparams for one env will most liklely not work for another, unless the new one is siginificantly easier. 

my code will solve all environments if hyperparams are done well!

DIffuculty ranking:
- Cartpole: instant solve (discrete) /n
- LunarLander: quick solve (3M ish) (discrete)
- BipedalWalker (normal): resonable solve (5-7Mish) (continuous)
- BipedalWalker (hardcore): frustratingly hard, expect progress at around 15M, solving the env at around 30M+ (Continuous)
- Humanoid (MuJoCo): Suprisingly learning to walk seems to be an easier task than hardcore bipedal (5M-10Mish) (continuous)
- HumanoidStandup (MuJoCo) definately hard, progress at 15M (attempted standups) maintained upright standing at 30M+ (continuous)
