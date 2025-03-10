// run PyFirmata on arduino

import pyfirmata
import time
import subprocess

board = pyfirmata.Arduino('COM3')

it = pyfirmata.util.Iterator(board)
it.start()

board.digital[3].mode = pyfirmata.INPUT
board.digital[4].mode = pyfirmata.INPUT
board.digital[5].mode = pyfirmata.INPUT

while True:
    but1 = board.digital[3].read()
    but2 = board.digital[4].read()
    but3 = board.digital[5].read()
    if but1 is True:
        print("button 1 pressed")
        subprocess.run(["python", "main.py", "--visualise", "--startpaused"])
    if but2 is True:
        print("button 2 pressed")
    if but3 is True:
        print("button 3 pressed")
    time.sleep(0.1)
