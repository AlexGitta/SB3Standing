import os

# Constants
PRINTREWARD = True
TARGET_HEIGHT = 1.63
SAVE_AT_STEP = 2005001
HEADLESS_EPOCHS = 10001
MAX_STEPS = 15000008


# Constants for reward calculation
TILT_THRESHOLD = 0.05 # Board tilt allowance
TILT_ALPHA = 3.0 # Steepness of tilt penalty curve
TILT_SCALE = 1.5 # Scale of overall tilt penalty
TILT_PENALTY_SCALE_LOW = 0.2 # Scale for penalty below threshold
SURVIVAL_REWARD_RATE = 0.001 # Reward / step for survival
SURVIVAL_START_STEP = 15 # Survival reward offset
HEIGHT_REWARD_SCALE = 1.5 # Scale for head height
HEIGHT_REWARD_OFFSET = -1.0 # Offset for height reward
ACTION_PENALTY_SCALE = 0.005 # Scale for exponential action penalty
EARLY_TERMINATION_PENALTY = -0.5 # Penalty for failing task
MAX_TILT_TERMINATION = 0.2 # Maximum board tilt (touching ground)
SUCCESS_REWARD = 1.0 # Reward for surviving 1000 steps
SUCCESS_STEPS = 1000 

TEMPERATURE = 0

EARLY_TERMINATION_HEIGHT = 1.2 # Height below which we terminate

