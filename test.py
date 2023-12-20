# Import the python library that connects to CoppeliaSim, the file is sim.py.
try:
    import sim
except:
    print('--------------------------------------------------------------')
    print('"sim.py" could not be imported. This means very probably that')
    print('either "sim.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "sim.py"')
    print('--------------------------------------------------------------')
    print('')

import time
from robot_control import robot


########################

     

# Actuator names
left_motor_name = "Pioneer_p3dx_leftMotor"
right_motor_name = "Pioneer_p3dx_rightMotor"

# Values to send  (rad/sec)
left_velocity = -1.0
right_velocity = 1.0


# Send the command!
print('Program started')
sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connect to CoppeliaSim

if clientID != -1:
    print('Connected to remote API server')

    # Now try to retrieve data in a blocking fashion (i.e. a service call):
    res, objs = sim.simxGetObjects(clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
    
    # Get handlers or actuators
    err_code, l_motor_handle = sim.simxGetObjectHandle(clientID, left_motor_name, sim.simx_opmode_blocking)
    err_code, r_motor_handle = sim.simxGetObjectHandle(clientID, right_motor_name, sim.simx_opmode_blocking)

    # Send the values!
    err_code = sim.simxSetJointTargetVelocity(clientID, l_motor_handle, left_velocity, sim.simx_opmode_streaming)
    err_code = sim.simxSetJointTargetVelocity(clientID, r_motor_handle, right_velocity, sim.simx_opmode_streaming)

    # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    sim.simxGetPingTime(clientID)

    # Now close the connection to CoppeliaSim:
    sim.simxFinish(clientID)
else:
    print('Failed connecting to remote API server')
print('Program ended')

###################
