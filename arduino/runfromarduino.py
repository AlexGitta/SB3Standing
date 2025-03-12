import serial
import time
import subprocess
import os

# Configure serial connection
ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)  # Allow time for Arduino to initialize

# Track active simulation process
active_process = None

# Create status file for communication with simulation
STATUS_FILE = "tilt_status.txt"

# Initialize the status file with "paused"
with open(STATUS_FILE, "w") as f:
    f.write("paused")

# Add debug information
print("Serial connection opened. Waiting for data...")

while True:
    # Check if active process has finished
    if active_process and active_process.poll() is not None:
        print("Simulation has finished")
        active_process = None
        # Reset status file when simulation ends
        with open(STATUS_FILE, "w") as f:
            f.write("paused")
    
    if ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
        # print(f"Raw data: {line}")  # Debug output to see what's being received
        
        try:
            # Parse comma-separated values: but1,but2,but3,gx,gy,gz,ax,ay,az
            values = line.split(',')
            if len(values) == 9:
                but1 = int(values[0])
                but2 = int(values[1])
                but3 = int(values[2]) 
                
                gyro_x = float(values[3])
                gyro_y = float(values[4])
                gyro_z = float(values[5])
                
                # Display sensor readings
                # print(f"Buttons: {but1}, {but2}, {but3}")
                print(f"Gyroscope: X={gyro_x:.4f}, Y={gyro_y:.4f}, Z={gyro_z:.4f}")

                # Detect board tilt and write status to file if simulation is running
                if active_process:
                    if abs(gyro_y) > 0:  # Use a small threshold to avoid noise
                        print("Board tilted - sending unpause signal")
                        with open(STATUS_FILE, "w") as f:
                            f.write("running")
                
                # Handle button presses only if no simulation is running
                if not active_process:
                    if but1 == 0:  # Button pressed (LOW signal)
                        print("Button 1 pressed, starting simulation with checkpoint 1")
                        # Reset status file to paused before starting
                        with open(STATUS_FILE, "w") as f:
                            f.write("paused")
                        active_process = subprocess.Popen(
                            ["python", "main.py", "--visualise", "--startpaused", "--checkpoint", "1"]
                        )
                    elif but2 == 0:
                        print("Button 2 pressed, starting simulation with checkpoint 2")
                        with open(STATUS_FILE, "w") as f:
                            f.write("paused")
                        active_process = subprocess.Popen(
                            ["python", "main.py", "--visualise", "--startpaused", "--checkpoint", "2"]
                        )
                    elif but3 == 0:
                        print("Button 3 pressed, starting simulation with checkpoint 3")
                        with open(STATUS_FILE, "w") as f:
                            f.write("paused")
                        active_process = subprocess.Popen(
                            ["python", "main.py", "--visualise", "--startpaused", "--checkpoint", "3"]
                        )
                else:
                    # A simulation is already running
                    if but1 == 0 or but2 == 0 or but3 == 0:
                        print("Button pressed but ignoring - simulation already running")
            else:
                print(f"Unexpected number of values: {len(values)}")
        except Exception as e:
            print(f"Error parsing data: {e}")
    
    time.sleep(0.01)  # Short delay