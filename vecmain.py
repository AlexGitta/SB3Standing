# PPO implementation for a humanoid balancing on a balance board
# using the MuJoCo physics engine and Stable Baselines3 library.
# The humanoid is trained to stand up and balance on the board, with various rewards and penalties applied based on its performance.
# The code includes a custom environment, a PPO agent, and a GUI for visualisation and interaction.
# This is the version of the code designed for training/testing. For the exhibition version, see the branch tagged "exhibition"
# By Alex Evans


import mujoco
import os
import glfw
import numpy as np
import imgui
import torch 
from imgui.integrations.glfw import GlfwRenderer
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from config.config import *
from environment.camera import InputState, center_camera_on_humanoid
import gym
from gym import spaces
import argparse

class StandupEnv(gym.Env):
    def __init__(self):
        super(StandupEnv, self).__init__()
        with open('resources/humanoidnew1.xml', 'r') as f: # Open xml file for humanoid model
            humanoid = f.read()
            self.model = mujoco.MjModel.from_xml_string(humanoid) # set model and data values 
            self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.model.nu,), dtype=np.float32) # Set action space 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.get_state()),), dtype=np.float32) # Set observation space using get_state
        self.episode_steps = 0 # Set basic tracking variables
        self.episode_reward = 0
        self.termination_penalty_applied = False
    
    def seed(self, seed=None): # Function to set a random initial set of values for an environment
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_state(self): # function to return current state of simulation environment
        return np.concatenate([
            self.data.qpos.flat,  # Relative joint positions
            self.data.qvel.flat,  # Relative joint velocities
            self.data.qfrc_actuator.flat, # Actuator forces
            self.data.cfrc_ext.flat,   # External forces
        ])

    def reset(self): # function to reset the environment on failure of task
        mujoco.mj_resetData(self.model, self.data) # Reset environment
        self.startheight = self.data.xpos[self.model.body('head').id][2]  # Tracking variables
        self.initial_qpos = self.data.qpos.copy()
        self.episode_steps = 0
        self.episode_reward = 0
        self.termination_penalty_applied = False
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, action): # function to take action and call reward function
        self.data.ctrl = np.clip(action, -1, 1)
        mujoco.mj_step(self.model, self.data) # Step the simulation
        next_state = self.get_state() # Get the state after step
        reward, done = self.calculate_reward() # Get reward
        self.episode_steps += 1
        return next_state, reward, done, {}

    def calculate_reward(self):

        # Get current state
        head_height = self.data.xpos[self.model.body('head').id][2]
        balance_board_id = self.model.body('balance_board').id
        board_quat = self.data.xquat[balance_board_id]
        tilt_side = abs(board_quat[1])
        tilt_fwd_back = abs(board_quat[2])
        total_tilt = tilt_side + tilt_fwd_back
        
        #  Board component - penalize board tilting
        if total_tilt < TILT_THRESHOLD:
            # Minimal linear penalty for small tilts
            balance_penalty = -total_tilt * (TILT_SCALE * TILT_PENALTY_SCALE_LOW)
        else:
            # Exponential penalty for larger tilts
            normalized_tilt = (total_tilt - TILT_THRESHOLD) / (1.0 - TILT_THRESHOLD)
            balance_penalty = -TILT_SCALE * (TILT_PENALTY_SCALE_LOW * TILT_THRESHOLD + (np.exp(TILT_ALPHA * normalized_tilt) - 1))

        # Survival reward for staying alive (grows over time, offset by 15 so agent isnt rewarded positvely immediately)
        survival_reward = (self.episode_steps - SURVIVAL_START_STEP) * SURVIVAL_REWARD_RATE # Basic reward that increases with survival time

        # Height reward based on head height (0.5 when at/above target height, minmum of -1 when below)
        height_factor = min(1.0, (head_height - EARLY_TERMINATION_HEIGHT) / (TARGET_HEIGHT - EARLY_TERMINATION_HEIGHT))
        height_reward = max(-1,height_factor - 0.5) 

        # Moderate penalty for excessive actions, using quadratic control cost
        action_penalty = -ACTION_PENALTY_SCALE * np.sum(np.square(self.data.ctrl))

        # Logic for termination due to falling/ board touching ground, or success (1000 steps reached)
        if ((head_height < EARLY_TERMINATION_HEIGHT) or tilt_side > MAX_TILT_TERMINATION) and not self.termination_penalty_applied:
            # Penalty for early termination
            termination_penalty = EARLY_TERMINATION_PENALTY
            self.termination_penalty_applied = True
        else:
            termination_penalty = 0.0

        if self.episode_steps > SUCCESS_STEPS:
            # Reward for successful completion
            success_reward = SUCCESS_REWARD
            done = True
        else:
            success_reward = 0.0

            # Early termination if fallen or balance board touches ground
            done = ((head_height < EARLY_TERMINATION_HEIGHT) or tilt_side > MAX_TILT_TERMINATION)

        # Combine components 
        reward = survival_reward + termination_penalty + success_reward + balance_penalty + action_penalty + height_reward
        
        # Accumulate episode reward
        self.episode_reward += reward

        # Reward inspection output
        if PRINTREWARD:
            print(f"Height: {height_reward:.2f}, " 
                f"Balance: {balance_penalty:.2f}, Action: {action_penalty:.2f}, "
                f"Survival: {survival_reward:.2f}, Step: {self.episode_steps}, Total: {reward:.2f}")
    
        return reward, done


class TensorboardCallback(BaseCallback):
    
    # custom callback for plotting additional values in tensorboard with multiple vectorized enviropnments
        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # For multiple environments, we log both the mean and individual values
            
            # Log values across all environments
            rewards = self.training_env.get_attr("episode_reward")
            steps = self.training_env.get_attr("episode_steps")

            
            if len(rewards) > 0:
                mean_reward = np.mean([r for r in rewards if r is not None])
                mean_steps = np.mean([s for s in steps if s is not None])
                
                
                self.logger.record("episode_total_reward/mean", mean_reward) # Log mean metrics
                self.logger.record("episode_steps/mean", mean_steps)
                
                for i, (reward, step) in enumerate(zip(rewards, steps)):
                    if reward is not None:
                        self.logger.record(f"episode_total_reward/env_{i}", reward) # Log metrics for each environment individually
                        self.logger.record(f"episode_steps/env_{i}", step)
            
            return True

def linear_schedule(initial_value: float, final_value: float): # Function to decrease a value linearly from x to y, using the progress left in the simulation
    def scheduler(progress_remaining: float) -> float: # Takes progress remaining as an argument, passed by MuJoCo
        return final_value + (initial_value - final_value) * (progress_remaining)
    
    return scheduler

def main():

    def make_env(rank, seed=0):  # Function to make environments
        def _init():
            env = StandupEnv()
            # Use a different seed for each environment
            env.seed(seed + rank)
            return env
        
        set_random_seed(seed)
        return _init
    
    parser = argparse.ArgumentParser(description="PPO Standing SB3")  # Argument parsing for various settings
    parser.add_argument('--visualise', action='store_true', help='Enable visualisation')
    parser.add_argument('--startpaused', action='store_true', help='Start paused')
    parser.add_argument('--checkpoint', type=int, default=1, help='Checkpoint (1-3)')

    # Commented out command-line hyperparameters, can be used  for rapid testing / hyperparameter tuning
    """
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--clip_param', type=float, default=CLIP_PARAM, help='Clip parameter')
    parser.add_argument('--ppo_epochs', type=int, default=PPO_EPOCHS, help='PPO epochs')
    parser.add_argument('--ent_coef', type=float, default=ENTROPY_COEF, help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=LOSS_COEF, help='Value function coefficient')
    """

    args = parser.parse_args()

    argVIS = args.visualise
    argSP = args.startpaused
    argCheck = args.checkpoint

    """
    argLR = args.learning_rate
    argCP = args.clip_param
    argEPOCH = args.ppo_epochs
    argENTCOEF = args.ent_coef
    argLOCOEF = args.vf_coef
    """

    os.makedirs('./vecpoints/', exist_ok=True)  # Set up checkpoints directory

    checkpoint_callback = CheckpointCallback( # Create callback to save checkpoints
        save_freq=SAVE_AT_STEP // 8, # Divide by no. of environments
        save_path='./vecpoints/',
        name_prefix='ppo_model',
        save_vecnormalize=True, # Save the vec normalisation
        verbose=1
    )
    
    initial_lr = 0.00006
    final_lr = 0.000003

    ppo_hyperparams = { # hyperparameters for PPO
            'learning_rate':  linear_schedule(initial_lr, final_lr), # learning rate schedule
            'clip_range': 0.2,
            'n_epochs': 4,  # Epochs per mini batch
            'ent_coef': 0.002,  # Coefficient for entropy loss
            'vf_coef': 0.7, # Value function coefficient
            'gamma': 0.995, # Discount Factor
            'gae_lambda': 0.95, # For generalised advantage estimation ( how much we value future advantage)
            'batch_size': 2048,
            'n_steps': 3072, # 3072 * no_envs is steps per update
            'policy_kwargs': dict(
                net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]), # Set neural net architectures, one for actor one for critic
                activation_fn=torch.nn.ReLU, # ReLu activation
                ortho_init=True,
            ),
            'normalize_advantage': True,
            'max_grad_norm': 0.2, # Gradient clipping
    }

    if argVIS:
        # Visualisation mode
        env = DummyVecEnv([lambda: StandupEnv()])  # Single env for visualization
        env = VecNormalize(env, norm_obs=True, norm_reward=True)    # Normalize environment (rewards between fixed range)
        model = PPO('MlpPolicy', env, verbose=1, **ppo_hyperparams) # Load model with hyperparameters
        input_state = InputState() # Initialize input state for camera control
        # Initialize GLFW and create window
        glfw.init()
        window = glfw.create_window(1200, 900, "Standup Task", None, None) # Create window for rendering
        glfw.make_context_current(window) 
        glfw.swap_interval(1) 

        # Setup scene
        camera = mujoco.MjvCamera() # Initialize camera 
        option = mujoco.MjvOption()
        scene = mujoco.MjvScene(env.get_attr('model')[0], maxgeom=100000) # Initialize scene
        context = mujoco.MjrContext(env.get_attr('model')[0], mujoco.mjtFontScale.mjFONTSCALE_150) # Initialize context for rendering

        # Set default camera and options
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 6.0
        camera.azimuth = 180
        camera.elevation = -15
        mujoco.mjv_defaultOption(option)


        # Initialize state
        state = env.reset()
        paused = argSP 
        reward = 0

        def handle_input(window, input_state, camera): # Function to handle input for camera control and other settings
            def mouse_button_callback(window, button, action, mods):  # Mouse button callback for camera control
                if button == glfw.MOUSE_BUTTON_LEFT:
                    input_state.left_down = action == glfw.PRESS
                    if input_state.left_down:
                        input_state.last_x, input_state.last_y = glfw.get_cursor_pos(window) # Get initial mouse position 
                elif button == glfw.MOUSE_BUTTON_RIGHT:
                    input_state.right_down = action == glfw.PRESS

            def mouse_move_callback(window, xpos, ypos):
                input_state.mouse_x = xpos
                input_state.mouse_y = ypos

                if input_state.left_down and input_state.camera_mode:  # Check if Alt key is pressed and mouse is down
                    dx = xpos - input_state.last_x # Get change in mouse position
                    dy = ypos - input_state.last_y
                    camera.azimuth += dx * 0.5
                    camera.elevation = np.clip(camera.elevation - dy * 0.5, -90, 90) # Update camera azimuth and elevation
                    input_state.last_x = xpos
                    input_state.last_y = ypos

            def keyboard_callback(window, key, scancode, action, mods): # Keyboard callback for various actions
                nonlocal paused
                if action == glfw.PRESS: 
                    if key == glfw.KEY_ESCAPE: # Escape key to exit
                        glfw.set_window_should_close(window, True)
                    elif key == glfw.KEY_SPACE: # Space key to pause/unpause simulation
                        paused = not paused

                # Track Alt key for camera mode
                input_state.camera_mode = (mods & glfw.MOD_ALT)

                # Camera azimuth control
                if key in [glfw.KEY_LEFT, glfw.KEY_RIGHT]: # Left/Right arrow keys to rotate camera
                    if action == glfw.PRESS or action == glfw.REPEAT:
                        if key == glfw.KEY_LEFT:
                            camera.azimuth = (camera.azimuth + 2) % 360
                        elif key == glfw.KEY_RIGHT:
                            camera.azimuth = (camera.azimuth - 2) % 360

            glfw.set_mouse_button_callback(window, mouse_button_callback) # Set mouse button callback
            glfw.set_cursor_pos_callback(window, mouse_move_callback) # Set mouse move callback
            glfw.set_key_callback(window, keyboard_callback) # Set keyboard callback

        # Initialize ImGui
        imgui.create_context()
        impl = GlfwRenderer(window) # Initialize ImGui renderer
        handle_input(window, input_state, camera) # Set input handling for camera control / other settings

        # Add checkpoint list state
        checkpoint_files = [] # List of checkpoint files
        selected_checkpoint = -1

        def update_checkpoint_list(): # Function to update the checkpoint list
            nonlocal checkpoint_files
            checkpoint_files = glob.glob(os.path.join("vecpoints", "*.zip")) # Get all checkpoint files
            checkpoint_files.sort(key=os.path.getctime, reverse=True)  # Sort by creation time


        update_checkpoint_list()

        reward_index = 0
        reward_min = float('0')
        reward_max = float('0')

        while not glfw.window_should_close(window): # Main loop for rendering and simulation
            if not paused:
                # Get action from policy
                action, _ = model.predict(state)
                state, reward, done, _ = env.step(action)
                if PRINTREWARD: # Print for debugging
                    print(f"Normalized Reward: ", reward)
                if done:
                    state = env.reset()

            center_camera_on_humanoid(camera, env.get_attr('data')[0], env.get_attr('model')[0]) # Center camera on humanoid using function from camera.py
                
            # Start ImGui frame
            impl.process_inputs()
            imgui.new_frame()

            # Create simulation stats window
            imgui.set_next_window_position(10, 10, imgui.ONCE)
            imgui.begin("Simulation Stats", True)
            imgui.set_window_size(200, 100, imgui.FIRST_USE_EVER)
           # imgui.text(f"Steps: {env.get_attr('total_steps')[0]}")
            imgui.text(f"Current Reward: {reward}")
            imgui.end()

            # Create checkpoint selector window
            imgui.set_next_window_position(10, 100, imgui.ONCE)
            imgui.begin("Checkpoint Selector", True)
            imgui.set_window_size(300, 200, imgui.FIRST_USE_EVER)

             # Create window with controls and instructions
            imgui.set_next_window_position(10, 300, imgui.ONCE)
            imgui.begin('Controls', True)
            imgui.text("Hold Alt + Left Mouse to rotate camera")
            imgui.text("Press Space key to pause/unpause")
            imgui.end()

            # Checkpoint list selector button functionality
            if imgui.button("Refresh List"):
                update_checkpoint_list()

            imgui.separator()

            # Display checkpoint files in a list
            for i, checkpoint_path in enumerate(checkpoint_files):
                filename = os.path.basename(checkpoint_path)
                if imgui.selectable(filename, selected_checkpoint == i)[0]:
                    selected_checkpoint = i
                    model = PPO.load(checkpoint_path, env=env)

            imgui.end()

            # Render ImGui
            imgui.render()

            # Mujoco rendering code
            viewport = mujoco.MjrRect(0, 0, 0, 0)
            viewport.width, viewport.height = glfw.get_framebuffer_size(window)
            
            # Set options for rendering
            mujoco.mjv_updateScene(
                env.get_attr('model')[0],
                env.get_attr('data')[0],
                option,
                None,
                camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene)
            mujoco.mjr_render(viewport, scene, context)

            impl.render(imgui.get_draw_data())

            glfw.swap_buffers(window)
            glfw.poll_events()


        # Cleanup
        impl.shutdown()
        glfw.terminate()

    else:
        # Headless training mode
        env = SubprocVecEnv([make_env(i, seed=42) for i in range(8)])  # 8 parallel envs for training
        env = VecNormalize(env, norm_obs=True, norm_reward=True) #  normalize environment (rewards between fixed range)
        log_path = "./tensorboard/" # Set up tensorboard logging directory
        model = PPO('MlpPolicy', env, verbose=1, **ppo_hyperparams, tensorboard_log=log_path) # Load model with hyperparameters and tensorboard logging
        tensorboard_callback = TensorboardCallback() # Create tensorboard callback
        model.learn(total_timesteps=MAX_STEPS, callback=[checkpoint_callback, tensorboard_callback])  # Train model with callbacks
        env.save("vec_normalize.pkl") # Save the VecNormalize object

if __name__ == "__main__":
    main()
