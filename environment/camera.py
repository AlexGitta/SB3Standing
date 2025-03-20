import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
import numpy as np

def center_camera_on_humanoid(camera, data, model):
    # Get torso position
    torso_pos = data.xpos[model.body('root').id]
    
    # Update camera lookat to track torso
    camera.lookat[0] = torso_pos[0]  # x
    camera.lookat[1] = torso_pos[1]  # y
    camera.lookat[2] = torso_pos[2]  # z

class InputState:
    def __init__(self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.left_down = False
        self.right_down = False
        self.camera_mode = False  # Alt held
        self.last_x = 0
        self.last_y = 0