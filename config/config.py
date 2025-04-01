import os

# Constants
PRINTREWARD = True 
TARGET_HEIGHT = 1.63
SAVE_AT_STEP = 2000001
HEADLESS_EPOCHS = 10001
MAX_STEPS = 20000001
PRINT_EPOCHS=10
PLOT_STEPS = 500
BUFFER_CAP = 10000

# Hyperparameters
TRAIN_INTERVAL = 128
UPH_COST_WEIGHT = 0.8
CTRL_COST_WEIGHT = 0.001
HEALTH_COST_WEIGHT = 0.001
FEET_COST_WEIGHT = 0.2
BALANCE_COST_WEIGHT = 2
HEIGHT_BONUS = 8
LEARNING_RATE = 0.0005

# PPO Hyperparameters
CLIP_PARAM = 0.2 # increase if stuck in local minima
PPO_EPOCHS = 10 # increase if training too slowly
LOSS_COEF = 0.5 # higher = emphasis on value function, lower = emphasis on policy improvement
ENTROPY_COEF = 0.001 # increase for more exploration

EARLY_TERMINATION_HEIGHT = 1

