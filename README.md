# Ecosystem-Parallel

###### By Bartłomiej Tarcholik

This is a PettingZoo Parallel Environment made for training 2 rivaling agent types - prey feeding on plants and predators feeding on prey.

By using the train_and_evaluate.py file you can train and see how the models perform. Change the parameters to whatever you wish. An actor-critic learning method is recommended, provided example uses PPO.
A pre-trained model has been provided although it's functionality is limited.

## Action and Observation spaces

### Action space

The action space for the agents is a Discrete(7) space where the following actions are possible:

- 0 - move +1 in Y axis
- 1 - move -1 in Y axis
- 2 - move +1 in X axis
- 3 - move -1 in X axis
- 4 - move +1 in Depth axis
- 5 - move -1 in Depth axis
- 6 - perform Eat action

Due to limitations of StableBaselines3 in Parallel Environments, illegal action mask is not available and is instead simulated by penalizing illegal actions.

### Observation space

Observation space is a Discrete(23) space with -1.0 to 1.0 range where the following observations are provided:

- Agent data:
  - Agent type (-1 for shark, 1 for fish)
  - Agent world coordinates
  - Remaining food and health points
- Terrain data:
  - Terrain beneath and around the agent, 1 unit to each side
- Data for closest and 2nd closest fish, shark and food
  - Distance
  - Direction represented by a single value (corresponds to action directions)

## Training

Training has been performed using the PPO algorithm provided by StableBaselines3 library. The model has been trained with the following parameters:

- Learning rate: 1e-3
- Gamma: 0.95
- Steps: 100 000 000
- Batch size: 256

Results of training point to the issue of having a single model taking care of both predator and prey agents as well as lack of illegal action mask and properly balanced rewards and penalties.

## Personal usage

Install required packages using requirements.txt

In order to use the training by yourself, you can use the train_and_evaluate.py file. Uncomment either the train() or eval() functions by calling them from **main()** and launch the training or evaluation.

You can modify ecosystem parameters as well - rewards, penalties, amount of agents of each type etc.
