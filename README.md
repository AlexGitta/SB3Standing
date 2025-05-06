 # SB3Standing
Standing (balance board) program implemented with Stable Baselines 3. This is the exhibition version of the project, with code that enables a user to compete with the agent via an MPU6050 Gyro mounted to a physical balance board. The user can also select checkpoints using 3 buttons. All data is sent through an Arduino via serial.








https://github.com/user-attachments/assets/58820c0f-c2ad-4ed5-8612-1c10d38cb337

This project is a reinforcement learning system where humanoid agents learn to maintain balance on a dynamic balance board in the MuJoCo physics simulation, using a carefully crafted PPO implementation. I combined sophisticated machine learning with a physical human-in-the-loop component, allowing users to compete with the AI agent on a real balance board fitted with gyroscope sensors, creating an engaging comparative study of human versus machine learning processes.

Requirements -  
Python 3.10
Imgui 2.0
MuJoCo 3.2.6  
GLFW 2.8.0  
Numpy 2.2.0  
PyTorch 2.5.1  
Stable Baselines 3 2.5.0  
OpenAI Gym 0.26.2  
TensorBoard 2.19.0  

To run the simulation, run
`python vecmain.py`  
To use the Arduino functionality, connect your Arduino UNO, mpu6050 and 3 buttons via serial and run `arduinocode.ino` on the Arduino.   
Then run `arduinoreciever.py` in order to visualise the gyroscope pitch reading, as well as change the loaded checkpoint using the buttons.  

Use space to pause/unpause, and ALT + mouse to move the camera.

Please note, if you want to use this version of the project, you will need the main version of the project in order to train agents headlessly.
