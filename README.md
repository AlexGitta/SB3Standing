# SB3Standing
Standing (balance board) program implemented with Stable Baselines 3. This is the version dedicated to training and visualising agents. If you would like to see the code designed for an exhibition setup, with a gyroscope and buttons communicating via serial, it is in a seperate branch called exhibition.








https://github.com/user-attachments/assets/58820c0f-c2ad-4ed5-8612-1c10d38cb337

This project is a reinforcement learning system where humanoid agents learn to maintain balance on a dynamic balance board in the MuJoCo physics simulation, using a carefully crafted PPO implementation. I combined sophisticated machine learning with a physical human-in-the-loop component, allowing users to compete with the AI agent on a real balance board fitted with gyroscope sensors, creating an engaging comparative study of human versus machine learning processes.

Requirements -  
MuJoCo 3.2.6  
GLFW 2.8.0  
Numpy 2.2.0  
PyTorch 2.5.1  
Stable Baselines 3 2.5.0  
OpenAI Gym 0.26.2  
TensorBoard 2.19.0  

To train an agent, run
`python vecmain.py`  
To configure the total number of steps to train for, how often to save a checkpoint and other parameters, edit `config.py`.  
To visualise the environment, run
`python vecmain.py --visualise`  
Use `--startpaused`to start the environment paused.  
Once you have trained an agent, open the visualisation window, select a checkpoint and unpause the simulation to see the results of the agent's learned policy.  
Use space to pause/unpause, and ALT + mouse to move the camera.
