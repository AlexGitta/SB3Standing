import mujoco
import os
import glfw
import numpy as np
import imgui
import torch 
from imgui.integrations.glfw import GlfwRenderer
import matplotlib.pyplot as plt
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed, get_linear_fn
from config.config import *
from environment.camera import InputState, center_camera_on_humanoid
import gym
from gym import spaces
import argparse

class StandupEnv(gym.Env):
    def __init__(self):
        super(StandupEnv, self).__init__()
        with open('resources/humanoidnew1fixed.xml', 'r') as f: # Open xml file for humanoid model
            humanoid = f.read()
            self.model = mujoco.MjModel.from_xml_string(humanoid) # set model and data values 
            self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.get_state()),), dtype=np.float32)
        print(f"Observation space: {self.observation_space.shape}, Action space: {self.action_space.shape}")
        self.episode_steps = 0
        self.episode_reward = 0
        self.termination_penalty_applied = False
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_state(self): # function to return current state of simulation environment
        return np.concatenate([
            self.data.qpos.flat,  
            self.data.qvel.flat,
           # self.data.cinert.flat,
           # self.data.cvel.flat,
            self.data.qfrc_actuator.flat,
            self.data.cfrc_ext.flat,
        ])

    def reset(self): # function to reset the environment on failure of task
        mujoco.mj_resetData(self.model, self.data)
        self.startheight = self.data.xpos[self.model.body('head').id][2]
        #starting_pose = self.model.key_qpos[0]
        #self.data.qpos[:] = starting_pose
        self.initial_qpos = self.data.qpos.copy()
        self.episode_steps = 0
        self.episode_reward = 0
        self.termination_penalty_applied = False
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, action): # function to take action and call reward function
        self.data.ctrl = np.clip(action, -1, 1)
        mujoco.mj_step(self.model, self.data)
        next_state = self.get_state()
        reward, done = self.calculate_reward()
        self.episode_steps += 1
        return next_state, reward, done, {}

    def calculate_reward(self):

        if self.episode_steps == 0:
            # Apply small random force to create initial instability 
            # This improves the stochasticity of the environment and helps the agent learn to balance
            random_force = self.np_random.uniform(-10, 10, size=3)
            self.data.xfrc_applied[self.model.body('torso').id, :3] = random_force

        head_height = self.data.xpos[self.model.body('head').id][2]
        
        #  Balance component - penalize board tilting
       # balance_board_id = self.model.body('balance_board').id
       # board_quat = self.data.xquat[balance_board_id]
        # Extract side-to-side and forward-backward tilt components
      #  tilt_side = abs(board_quat[1]) * 10.0
       # tilt_fwd_back = abs(board_quat[2]) * 10.0
       # balance_penalty = -(tilt_side + tilt_fwd_back)
        
        #  Stability component - reward minimal velocities
        com_velocity = np.linalg.norm(self.data.qvel[:3])  # Center of mass velocity
        stability_reward = -0.5 * com_velocity
 
        joint_qpos = self.data.qpos
        initial_joint_qpos = self.initial_qpos    
        all_indices = list(range(len(joint_qpos)))

        non_arm_indices = list(range(len(joint_qpos) - 6))  # Exclude the last 6 joints (arm joints)

              # Calculate posture differences
        raw_diff = np.abs(joint_qpos[all_indices] - initial_joint_qpos[all_indices])
        
        # Exponential posture penalty calculation
        # Parameters for the exponential function
        threshold = 0.025  # Small deviations below this are minimally penalized
        alpha = 10      # Controls how quickly the penalty grows
        scale = 0.0003     # Overall scale of the penalty
        
        # Apply threshold and calculate exponential penalty
        # For differences below threshold, apply minimal penalty
        mask_small = raw_diff < threshold
        mask_large = ~mask_small
        
        # Small deviations: apply minimal linear penalty
        small_penalty = raw_diff[mask_small] * (scale/5)
        
        # Large deviations: apply exponential penalty
        normalized_diff = (raw_diff[mask_large] - threshold) / (1.0 - threshold)  # Normalize to 0-1 range
        large_penalty = scale * (np.exp(alpha * normalized_diff) - 1)
        
        # Combine penalties
        combined_penalty = np.zeros_like(raw_diff)
        combined_penalty[mask_small] = small_penalty
        combined_penalty[mask_large] = large_penalty
        
        # Calculate total posture penalty
        posture_penalty = max(-1, -np.sum(combined_penalty))

       # Primary reward for staying alive (grows over time)

        survival_reward = ((self.episode_steps -15 ) * 0.001) # Basic reward that increases with survival time

        # Height component (positive reward when at target height, drops off as height decreases)
        height_factor = min(1.0, (head_height - EARLY_TERMINATION_HEIGHT) / (TARGET_HEIGHT - EARLY_TERMINATION_HEIGHT))
        height_reward = 1.0 * height_factor  # Scales from 0 to 1 based on height

        # Moderate penalty for excessive actions
        action_penalty = -0.0025 * np.sum(np.square(self.data.ctrl))

        # Strong termination penalty
        if (head_height < EARLY_TERMINATION_HEIGHT) and not self.termination_penalty_applied:
            termination_penalty = -0.5
            self.termination_penalty_applied = True
        else:
            termination_penalty = 0.0

        if self.episode_steps > 1000:
            success_reward = 1.0
            done = True
        else:
            success_reward = 0.0
            # Early termination if fallen
            done = head_height < EARLY_TERMINATION_HEIGHT

        # Combine components 
       # reward = survival_reward + height_reward + posture_penalty + action_penalty + termination_penalty 
        #reward = posture_penalty + survival_reward + termination_penalty + success_reward
        reward = posture_penalty + survival_reward + termination_penalty
       
        # Accumulate episode reward
        self.episode_reward += reward
        
        # Early termination if fallen

        
        
        if PRINTREWARD:
            print(f"Height: {height_reward:.2f}, " 
                f"Stability: {stability_reward:.2f}, Action: {action_penalty:.2f}, "
                f"Survival: {survival_reward:.2f}, Posture: {posture_penalty:.2f}, Step: {self.episode_steps}, Total: {reward:.2f}")
    
        return reward, done


class TensorboardCallback(BaseCallback):
    
    # custom callback for plotting additional values in tensorboard with multiple vectorized enviropnments
        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # For multiple environments, we can log either the mean or individual values
            
            # Log values across all environments
            rewards = self.training_env.get_attr("episode_reward")
            steps = self.training_env.get_attr("episode_steps")

            
            if len(rewards) > 0:
                mean_reward = np.mean([r for r in rewards if r is not None])
                mean_steps = np.mean([s for s in steps if s is not None])
                
                
                self.logger.record("episode_total_reward/mean", mean_reward)
                self.logger.record("episode_steps/mean", mean_steps)
                
                for i, (reward, step) in enumerate(zip(rewards, steps)):
                    if reward is not None:
                        self.logger.record(f"episode_total_reward/env_{i}", reward)
                        self.logger.record(f"episode_steps/env_{i}", step)
            
            return True

def linear_schedule(initial_value: float, final_value: float):
    def scheduler(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * (progress_remaining)
    
    return scheduler
    
def exponential_schedule(initial_value: float, final_value: float):
    def scheduler(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * (progress_remaining ** 2)
    
    return scheduler

def main():

    def make_env(rank, seed=0):
        def _init():
            env = StandupEnv()
            # Use a different seed for each environment
            env.seed(seed + rank)
            return env
        
        set_random_seed(seed)
        return _init
    
    parser = argparse.ArgumentParser(description="PPO Standing SB3")
    parser.add_argument('--visualise', action='store_true', help='Enable visualisation')
    parser.add_argument('--startpaused', action='store_true', help='Start paused')
    parser.add_argument('--checkpoint', type=int, default=1, help='Checkpoint (1-3)')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--clip_param', type=float, default=CLIP_PARAM, help='Clip parameter')
    parser.add_argument('--ppo_epochs', type=int, default=PPO_EPOCHS, help='PPO epochs')
    parser.add_argument('--ent_coef', type=float, default=ENTROPY_COEF, help='Entropy coefficient')
    parser.add_argument('--vf_coef', type=float, default=LOSS_COEF, help='Value function coefficient')
    args = parser.parse_args()

    argVIS = args.visualise
    argSP = args.startpaused
    argCheck = args.checkpoint
    argLR = args.learning_rate
    argCP = args.clip_param
    argEPOCH = args.ppo_epochs
    argENTCOEF = args.ent_coef
    argLOCOEF = args.vf_coef

    os.makedirs('./vecpoints/', exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_AT_STEP // 6,
        save_path='./vecpoints/',
        name_prefix='ppo_model',
      #  save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )
    
    initial_lr = 0.00005
    final_lr = 0.000003
    initial_clip = 0.2
    final_clip = 0.04
    initial_ent = 0.004
    final_ent = 0.0

    ppo_hyperparams = { # hyperparameters for PPO
            'learning_rate':  initial_lr, # learning rate schedule
            'clip_range': initial_clip,
            'n_epochs': 4,  
            'ent_coef': initial_ent,  
            'vf_coef': 0.7,
            'gamma': 0.995,
            'gae_lambda': 0.95,
            'batch_size': 2048,
            'n_steps': 3072, 
            'policy_kwargs': dict(
                net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                activation_fn=torch.nn.LeakyReLU,
                ortho_init=True,
            ),
            'normalize_advantage': True,
            'max_grad_norm': 0.3,
    }

    if argVIS:
        env = DummyVecEnv([lambda: StandupEnv()])  # Single env for visualization
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        model = PPO('MlpPolicy', env, verbose=1, **ppo_hyperparams)
        input_state = InputState()
        # Initialize GLFW and create window
        glfw.init()
        window = glfw.create_window(1200, 900, "Standup Task", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        # Setup scene with original camera setup
        camera = mujoco.MjvCamera()
        option = mujoco.MjvOption()
        scene = mujoco.MjvScene(env.get_attr('model')[0], maxgeom=100000)
        context = mujoco.MjrContext(env.get_attr('model')[0], mujoco.mjtFontScale.mjFONTSCALE_150)

        # Set default camera and options with adjusted angle
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 6.0
        camera.azimuth = 90
        camera.elevation = -15
        mujoco.mjv_defaultOption(option)


        # Initialize state
        state = env.reset()
        paused = argSP 
        reward = 0

        def handle_input(window, input_state, camera):
            def mouse_button_callback(window, button, action, mods):
                if button == glfw.MOUSE_BUTTON_LEFT:
                    input_state.left_down = action == glfw.PRESS
                    if input_state.left_down:
                        input_state.last_x, input_state.last_y = glfw.get_cursor_pos(window)
                elif button == glfw.MOUSE_BUTTON_RIGHT:
                    input_state.right_down = action == glfw.PRESS

            def mouse_move_callback(window, xpos, ypos):
                input_state.mouse_x = xpos
                input_state.mouse_y = ypos

                if input_state.left_down and input_state.camera_mode:
                    dx = xpos - input_state.last_x
                    dy = ypos - input_state.last_y
                    camera.azimuth += dx * 0.5
                    camera.elevation = np.clip(camera.elevation - dy * 0.5, -90, 90)
                    input_state.last_x = xpos
                    input_state.last_y = ypos

            def keyboard_callback(window, key, scancode, action, mods):
                nonlocal paused
                if action == glfw.PRESS:
                    if key == glfw.KEY_ESCAPE:
                        glfw.set_window_should_close(window, True)
                    elif key == glfw.KEY_SPACE:
                        paused = not paused

                # Track Alt key for camera mode
                input_state.camera_mode = (mods & glfw.MOD_ALT)

                # Camera azimuth control
                if key in [glfw.KEY_LEFT, glfw.KEY_RIGHT]:
                    if action == glfw.PRESS or action == glfw.REPEAT:
                        if key == glfw.KEY_LEFT:
                            camera.azimuth = (camera.azimuth + 2) % 360
                        elif key == glfw.KEY_RIGHT:
                            camera.azimuth = (camera.azimuth - 2) % 360

            glfw.set_mouse_button_callback(window, mouse_button_callback)
            glfw.set_cursor_pos_callback(window, mouse_move_callback)
            glfw.set_key_callback(window, keyboard_callback)

        # Initialize ImGui
        imgui.create_context()
        impl = GlfwRenderer(window)
        handle_input(window, input_state, camera)

        # Add checkpoint list state
        checkpoint_files = []
        selected_checkpoint = -1

        def update_checkpoint_list():
            nonlocal checkpoint_files
            checkpoint_files = glob.glob(os.path.join("vecpoints", "*.zip"))
            checkpoint_files.sort(key=os.path.getctime, reverse=True)  # Sort by creation time

        update_checkpoint_list()

        reward_buffer = np.full(PLOT_STEPS, np.nan, dtype=np.float32)
        reward_index = 0
        reward_min = float('0')
        reward_max = float('0')

        while not glfw.window_should_close(window):
            if not paused:
                # Get action from policy
                action, _ = model.predict(state)
                state, reward, done, _ = env.step(action)
                reward_buffer[reward_index] = reward
                reward_index = (reward_index + 1) % PLOT_STEPS
                reward_min = min(reward_min, np.nanmin(reward_buffer))
                reward_max = max(reward_max, np.nanmax(reward_buffer))
                if PRINTREWARD:
                    print(f"Normalized Reward: ", reward)
                if done:
                    state = env.reset()

            center_camera_on_humanoid(camera, env.get_attr('data')[0], env.get_attr('model')[0])
            # Check tilt_status.txt file to see if we should unpause
            try:
                with open("tilt_status.txt", "r") as f:
                    status = f.read().strip()
                    if status == "running" and paused:
                        print("Unpausing simulation due to board tilt")
                        paused = False
            except Exception as e:
                print(f"Error reading status file: {e}")
                
            # Start ImGui frame
            impl.process_inputs()
            imgui.new_frame()

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

            imgui.set_next_window_position(10, 300, imgui.ONCE)
            imgui.begin('Controls', True)
            imgui.text("Hold Alt + Left Mouse to rotate camera")
            imgui.text("Press Space key to pause/unpause")
            imgui.end()

            imgui.set_next_window_position(10, 500, imgui.ONCE)
            imgui.begin("Reward History", True)
            imgui.set_window_size(300, 200, imgui.FIRST_USE_EVER)

            # Plot the rewards
            if len(reward_buffer) > 1:
                imgui.plot_lines("##rewards",
                                 reward_buffer,
                                 graph_size=(285, 150),
                                 scale_min=reward_min,
                                 scale_max=reward_max)
                imgui.text(f"Min: {reward_min:.2f} Max: {reward_max:.2f}")
            imgui.end()

            if imgui.button("Refresh List"):
                update_checkpoint_list()

            imgui.separator()

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
        env = SubprocVecEnv([make_env(i, seed=42) for i in range(8)])  # 8 parallel envs for training
        env = VecNormalize(env, norm_obs=True, norm_reward=True) #  normalize environment (rewards between fixed range)
        log_path = "./tensorboard/"
        model = PPO('MlpPolicy', env, verbose=1, **ppo_hyperparams, tensorboard_log=log_path)
        tensorboard_callback = TensorboardCallback()
        model.learn(total_timesteps=MAX_STEPS, callback=[checkpoint_callback, tensorboard_callback])
        env.save("vec_normalize.pkl")

if __name__ == "__main__":
    main()