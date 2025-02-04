import mujoco
import glfw
import torch
import numpy as np
import imgui
from imgui.integrations.glfw import GlfwRenderer
import matplotlib.pyplot as plt
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from config.config import *
from environment.camera import InputState, center_camera_on_humanoid
import gym
from gym import spaces

class StandupEnv(gym.Env):
    def __init__(self):
        super(StandupEnv, self).__init__()
        with open('resources/humanoidnew.xml', 'r') as f:
            humanoid = f.read()
            self.model = mujoco.MjModel.from_xml_string(humanoid)
            self.data = mujoco.MjData(self.model)


        self.action_space = spaces.Box(low=-1, high=1, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.get_state()),), dtype=np.float32)

    def get_state(self):
        return np.concatenate([
            self.data.qpos.flat[2:],  # Skip root x/y coordinates
            self.data.qvel.flat,
            self.data.cinert.flat,
            self.data.cvel.flat,
            self.data.qfrc_actuator.flat,
            self.data.cfrc_ext.flat,
        ])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, action):
        self.data.ctrl = np.clip(action, -1, 1)
        mujoco.mj_step(self.model, self.data)
        next_state = self.get_state()
        reward, done = self.calculate_reward()
        return next_state, reward, done, {}

    def calculate_reward(self):
        head_height = self.data.xpos[self.model.body('head').id][2]
        reward = head_height  # Simplified reward for demonstration
        done = head_height < EARLY_TERMINATION_HEIGHT
        return reward, done

    def render(self, mode='human'):
        pass

    def close(self):
        pass

def main():
    env = DummyVecEnv([lambda: StandupEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    checkpoint_callback = CheckpointCallback(save_freq=SAVE_AT_STEP, save_path='./checkpoints/', name_prefix='ppo_model')
    if VISUALISE:
        input_state = InputState()
        # Initialize GLFW and create window
        episode = 1
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
        camera.azimuth = 180
        camera.elevation = -20
        mujoco.mjv_defaultOption(option)

        # Initialize agent
        model = PPO('MlpPolicy', env, verbose=1)
        checkpoint_callback = CheckpointCallback(save_freq=SAVE_AT_STEP, save_path='./checkpoints/', name_prefix='ppo_model')

        # Initialize state
        state = env.reset()
        paused = False

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
            checkpoint_files = glob.glob(os.path.join("checkpoints", "*.zip"))
            checkpoint_files.sort(key=os.path.getctime, reverse=True)  # Sort by creation time

        update_checkpoint_list()

        reward_buffer = np.zeros(PLOT_STEPS, dtype=np.float32)
        reward_index = 0
        reward_min = float('inf')
        reward_max = float('-inf')

        while not glfw.window_should_close(window):
            if not paused:
                # Get action from policy
                action, _ = model.predict(state)
                state, reward, done, _ = env.step(action)
                reward_buffer[reward_index] = reward
                reward_index = (reward_index + 1) % PLOT_STEPS
                reward_min = min(reward_min, np.min(reward_buffer))
                reward_max = max(reward_max, np.max(reward_buffer))

                # Check episode end
                if done:
                    state = env.reset()

            center_camera_on_humanoid(camera, env.get_attr('data')[0], env.get_attr('model')[0])
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
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=MAX_STEPS, callback=checkpoint_callback)

if __name__ == "__main__":
    main()