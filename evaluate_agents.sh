LANG=en_US
python3 eval_agents.py --left ppo --right tour1 --render --trials 20

# ["baseline", "ppo", "ga", "cma", "random", "tour", "tour1", "tour2", "tour3", "tour4"]
# ppo here refers to ppo-self-play algorithm
# you can evaluate select any of the below algorithm to play against the other
# The Evaluation result printed out in the terminal is the evaluation for the right hand policy