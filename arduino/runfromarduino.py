import serial
import time
import subprocess

# Configure serial connection
ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)  # Allow time for Arduino to initialize

# Track active simulation process
active_process = None

# Add debug information
print("Serial connection opened. Waiting for data...")

while True:
    # Check if active process has finished
    if active_process and active_process.poll() is not None:
        print("Simulation has finished")
        active_process = None
    
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

                if(gyro_y > 0 or gyro_y < 0):
                    print("board tilted")
                
                # Handle button presses only if no simulation is running
                if not active_process:
                    if but1 == 0:  # Button pressed (LOW signal)
                        print("Button 1 pressed, starting simulation with checkpoint 1")
                        active_process = subprocess.Popen(
                            ["python", "main.py", "--visualise", "--startpaused", "--checkpoint", "1"]
                        )
                    elif but2 == 0:
                        print("Button 2 pressed, starting simulation with checkpoint 2")
                        active_process = subprocess.Popen(
                            ["python", "main.py", "--visualise", "--startpaused", "--checkpoint", "2"]
                        )
                    elif but3 == 0:
                        print("Button 3 pressed, starting simulation with checkpoint 3")
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
