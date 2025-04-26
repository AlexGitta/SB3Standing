# Code to recieve MPU gyro values from the arduino, as well as the 3 button presses
# Using the gyro pitch value, we show a representation of the orientation using a line and a colour change
# We also use the button presses to change a text file specifying checkpoints to load, as these are read in the exhibmain.py file

import time
import math
import sys
import serial
import re
import glfw
from OpenGL.GL import *
import numpy as np

STATUS_FILE = "checkpoint.txt"

class MPU6050SerialVisualizer:
    def __init__(self, port='/dev/ttyUSB0', baud_rate=9600):
        # Initialize GLFW
        if not glfw.init():
            print("Failed to initialize GLFW")
            sys.exit(1)
        
        self.width, self.height = 800, 400
        self.window = glfw.create_window(self.width, self.height, "Board tilt", None, None)
        if not self.window:
            print("Failed to create GLFW window")
            glfw.terminate()
            sys.exit(1)
            
        glfw.make_context_current(self.window)
        glfw.set_key_callback(self.window, self.key_callback)
        
        # Connect to Arduino via serial
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=1)
            print(f"Connected to Arduino on {port} at {baud_rate} baud")
            time.sleep(2)  # Give the connection time to establish
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            glfw.terminate()
            sys.exit(1)

        # Button state variables
        self.button1 = 0
        self.button2 = 0
        self.button3 = 0

        with open(STATUS_FILE, "w") as f:
            f.write("0")
            
        # Pitch angle variable
        self.pitch = 0.0
        
        # Setup viewport and projection
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # For timing/FPS control
        self.previous_time = time.time()
        
    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            
    def read_pitch_from_serial(self):
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
              #  print(f"Raw line from serial: {line}")  # Debugging line
                
                # Look for the ypr (yaw, pitch, roll) line from the Arduino output
                if line.startswith("ypr"):
                    # Parse the pitch value (second value in the ypr\t0.00\t0.00\t0.00 format)
                    values = line.split('\t')
                    if len(values) >= 3:
                        # In the Arduino code, the second value (index 2) is the pitch
                        self.pitch = float(values[2])

                # Look for button states
                elif line.startswith("buts"):
                    # Parse the button values (buts\t0\t0\t0 format)
                    values = line.split('\t')
                    if len(values) >= 3:
                        self.button1 = int(values[1])
                        self.button2 = int(values[2])
                        self.button3 = int(values[3])
                        # print(f"Buttons: {self.button1}, {self.button2}, {self.button3}")  # Debugging
                
                    
            except Exception as e:
                print(f"Error reading from serial: {e}")
        
        return self.pitch
        
    def draw_pitch_indicator(self, pitch_angle):
        # Clear the screen
        glClearColor(0.12, 0.12, 0.12, 1.0)  # Dark gray background
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Center of rotation
        center_x, center_y = self.width // 2, self.height // 2
        
        # Draw a stationary horizon line
        glColor3f(0.4, 0.4, 0.4)  # Gray color
        glBegin(GL_LINES)
        glVertex2f(50, center_y)
        glVertex2f(self.width - 50, center_y)
        glEnd()
        
        # Calculate the length of the indicator line
        length = 300
        
        # Calculate endpoints of the rotated line
        angle_rad = math.radians(self.pitch + 90)
        dx = length * math.sin(angle_rad)  
        dy = length * math.cos(angle_rad)  
        
        start_x = center_x - dx / 2
        start_y = center_y - dy / 2
        end_x = center_x + dx / 2
        end_y = center_y + dy / 2
        
        # Draw the pitch indicator line
        glLineWidth(6.0)
        glColor3f(0.0, 0.63, 1.0)  # Blue color
        glBegin(GL_LINES)
        glVertex2f(start_x, start_y)
        glVertex2f(end_x, end_y)
        glEnd()
        glLineWidth(1.0)
        
        # Draw a small circle at the center
        glColor3f(1.0, 0.2, 0.2)  # Red color
        self.draw_circle(center_x, center_y, 8)
        
        glRasterPos2f(self.width//2 - 80, 50)
        glColor3f(1.0, 1.0, 1.0)  # White color
        
    def draw_circle(self, x, y, radius): # helper method to draw a circle
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)  # Center
        num_segments = 30
        for i in range(num_segments + 1):
            angle = 2.0 * math.pi * i / num_segments
            glVertex2f(x + math.cos(angle) * radius, y + math.sin(angle) * radius)
        glEnd()

    def get_button_states(self):
        return (self.button1, self.button2, self.button3)
        
    def run(self):
        while not glfw.window_should_close(self.window):
            # Update pitch from serial
           # print("Reading pitch from serial...")
            pitch_angle = self.read_pitch_from_serial()

            # Get state of buttons
            button1, button2, button3 = self.get_button_states()

            # Update the text file based on button states, used for checkpoint loading
            if button1:
                with open(STATUS_FILE, "w") as f:
                    f.write("run0.zip")
            elif button2:
                with open(STATUS_FILE, "w") as f:
                    f.write("run1.zip")
            elif button3:
                with open(STATUS_FILE, "w") as f:
                    f.write("run2.zip")
            
         #   print(f"Pitch angle: {pitch_angle:.2f}°")  # Debugging line
            # Draw the visualization
            self.draw_pitch_indicator(pitch_angle)
            
        #    print("Drawing pitch indicator...")  # Debugging line
            # Update display
            glfw.swap_buffers(self.window)
            print("Updating display...")  # Debugging line
            
            # Poll for events
            glfw.poll_events()
            
            # Control frame rate (60 FPS)
            current_time = time.time()
            delta = current_time - self.previous_time
            sleep_time = max(1.0/60.0 - delta, 0)
            time.sleep(sleep_time)
            self.previous_time = current_time
            
        # Clean up
        self.ser.close()
        glfw.terminate()

if __name__ == "__main__":
    PORT = 'COM3'  
    
    BAUD_RATE = 9600 
    
    try:
        visualizer = MPU6050SerialVisualizer(port=PORT, baud_rate=BAUD_RATE)
        visualizer.run()
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        if glfw.get_current_context():
            glfw.terminate()
