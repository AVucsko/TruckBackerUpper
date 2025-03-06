
#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import time


# --------------------------------------------------------
# ------------       Define Classes       ----------------
# --------------------------------------------------------
class tractor:
    def __init__(self, x, y, x_prev, y_prev, yaw, yaw_prev, length, width, vel):
        self.x = x
        self.y = y
        self.x_prev = x_prev
        self.y_prev = y_prev
        self.yaw = yaw
        self.yaw_prev = yaw_prev
        self.length = length
        self.width = width
        self.vel = vel
		
class trailer:
    def __init__(self, x, y, x_prev, y_prev, yaw, yaw_prev, length, width, vel):
        self.x = x
        self.y = y
        self.x_prev = x_prev
        self.y_prev = y_prev
        self.yaw = yaw
        self.yaw_prev = yaw_prev
        self.length = length
        self.width = width
        self.vel = vel


class pid:
    def __init__(self, integ, deriv, error, error_prev):
        self.integ = integ
        self.deriv = deriv
        self.error = error
        self.error_prev = error_prev


class sm:
    def __init__(self, s_x, s_x_prev, sy, sy_prev, error_x1, error_x1_dot, error_y1, error_y1_dot, error_x2, error_x2_dot, error_y2, error_y2_dot, error_xd, error_xd_dot, error_yd, error_yd_dot):
        self.s_x = s_x
        self.s_x_prev = s_x_prev
        self.s_y = s_y
        self.s_y_prev = s_y_prev
        self.error_x1 = error_x1
        self.error_x1_dot = error_x1_dot
        self.error_y1 = error_y1
        self.error_y1_dot = error_y1_dot
        self.error_x2 = error_x2
        self.error_x2_dot = error_x2_dot
        self.error_y2 = error_y2
        self.error_y2_dot = error_y2_dot
        self.error_xd = error_xd
        self.error_xd_dot = error_xd_dot
        self.error_yd = error_yd
        self.error_yd_dot = error_yd_dot

class nn:
    def __init__(self, w):
        self.w1 = w
    def setup():
        nn.w = np.zeros((1,24))
        nn.prev_scen = 0


# --------------------------------------------------------
# ------------    Supporting Functions    ----------------
# --------------------------------------------------------

def rate_limit(delta, delta_prev, dt):
    # Currently limiting the steering command to 360 deg/s -- In line with rereax limiters at this speed
    if abs((abs(delta) - abs(delta_prev)))/dt >= 0.0628:
        if delta > delta_prev:
            delta = delta_prev + 0.0628
        else:
            delta = delta_prev - 0.0628
   
    return(delta)


# --------------------------------------------------------
# -----------------    Controllers    --------------------
# --------------------------------------------------------

def pid_controller(goal, dt):
    # PID Controller to Reach Desired Articulation Angle
    #P = -7
    #I = -3
    #D = -0.5
    P = -5
    I = 0
    D = 0
    
    # Define Errors
    pid.error = goal - (tractor.yaw - trailer1.yaw)
    pid.integ = pid.integ + pid.error * dt
    pid.deriv = (pid.error - pid.error_prev) / dt
    
    # Desired Steering Angle
    delta = pid.error * P + pid.integ * I + pid.deriv * D
    
    if delta < -0.52:
        delta = -0.52
    elif delta > 0.52:
        delta = 0.52

    pid.error_prev = pid.error
    return(delta)
    

def sm_controller(dt):
    # Trailer 2 - Trailer 1
    k1x, qx = 0.0, -0.0
    k1y, qy = -0.0, -0.0

    vel = -1
    L2 = 16.15
    theta_x = np.arcsin(-np.cos(trailer2.yaw)/(vel*L2) + k1x * trailer2.x/vel)
    theta_y = np.arcsin(-k1y*trailer2.yaw*L2/vel) + trailer2.yaw
    theta2 = theta_x + theta_y
    sm.s_x_prev = sm.s_x
    sm.s_y_prev = sm.s_y

    # If close to zero error use a simpler control scheme
    #if np.abs(tractor.x) < 0.1:
    #    theta2 = trailer2.yaw
        
    # Limit the articulation angle we can request
    if theta2 > 1.57:
        theta2 = 1.57
    elif theta2 < -1.57:
        theta2 = -1.57

    theta2 = theta2 + trailer2.yaw
    #if theta2 > 3.14:
    #    theta2 = theta2 - 6.28
    #print(trailer2.yaw, trailer1.yaw)

    # Theta 2 is the articulation angle needed between 2 and 1
    art21 = trailer1.yaw - trailer2.yaw
    k2y = 0.2
    booga = -k2y*(trailer1.yaw - theta2)*L2/vel
    if booga > 1:
        booga = 1  
    elif booga < -1:
        booga = -1
    
    theta1 = np.arcsin(booga) + trailer1.yaw
    if theta1 > 1.57:
        theta1 = 1.57
    elif theta1 < -1.57:
        theta1 = -1.57
    
    print(theta2,theta1)
    return(theta1)



def nn_controller():
    nn.w = ga.weights[ga.chrom,:]

    #Complete
    #Best Alltime Fit:  0.22923277349510707
    #Weights:  [-0.16050434  0.18051051 -0.96533591  0.23874888  0.31570426 -1.
    #-0.52896247  0.75347422  0.43679287  0.73438701 -0.347128    0.66018496
    #-0.51024424  0.25566677  0.5669994  -0.64711265  0.94448799  0.49281207
    #-0.09850194  0.65096077  0.93635807  0.04509822 -0.46002495  0.72295723]
	
	# Neurons
    z1 = np.tanh((tractor.yaw-trailer1.yaw) * nn.w[0])
    z2 = np.tanh((trailer1.yaw - trailer2.yaw) * nn.w[1])
    z3 = np.tanh((trailer2.x) * nn.w[2])
    z4 = np.tanh((trailer2.yaw) * nn.w[3])
    z5 = np.tanh(z1 * nn.w[4] + z2 * nn.w[5] + z3 * nn.w[6] + z4 * nn.w[7]) 
    z6 = np.tanh(z1 * nn.w[8] + z2 * nn.w[9] + z3 * nn.w[10] + z4 * nn.w[11]) 
    z7 = np.tanh(z1 * nn.w[12] + z2 * nn.w[13] + z3 * nn.w[14] + z4 * nn.w[15]) 
    z8 = np.tan(z5 * nn.w[16] + z6 * nn.w[17] + z7 * nn.w[18])
    z9 = np.tan(z5 * nn.w[19] + z6 * nn.w[20] + z7 * nn.w[21])
    z10 = np.tanh(z8*nn.w[22] + z9*nn.w[23])

    theta = z10*10 # Accidentally using z7 here worked in the past somehow
    if theta > 1.57:
        theta = 1.57
    elif theta < -1.57:
        theta = -1.57
	
    return(theta)


# --------------------------------------------------------
# -----------------  Kinematic Model  --------------------
# --------------------------------------------------------

def kin_model(delta, dt, vel):
    # Important to keep in mind x and y are flipped from normal

    # x - X error measured at kingpin (meters)
    # y - Y error measured at kingpin (meters)
    # psi_tract - Angle of tractor measured relative to y-axis (rad)
    # psi_trail - Angle of trailer measured relative to y-axis (rad)
    # delta - Steered wheel angle (rad)
    # dt - Time step (s)
    # vel = Tractor velocity (m/s)
    
    # Setup vehicle parameters
    L1c = 0         #meters
    #L_tract = 6     #meters
    #L_trail = 16.15 #meters
    #L_trailer2 = 2.6   #meters
    
    # Calculate articulation angles and apply limits
    art_angle = tractor.yaw - trailer1.yaw
    art_angle2 = trailer1.yaw - trailer2.yaw

    # Calculate Velocities
    try:
        #trailer1.vel = -(np.abs((trailer1.x**2 - trailer1.x_prev**2)) + np.abs((trailer1.y**2 - trailer1.y_prev**2)))**0.5
        #trailer2.vel = -(np.abs((trailer2.x**2 - trailer2.x_prev**2)) + np.abs((trailer2.y**2 - trailer2.y_prev**2)))**0.5
        trailer1.vel = -(np.abs((trailer1.x**2 + trailer1.y**2) - (trailer1.x_prev**2 + trailer1.y_prev**2)))**0.5
        trailer2.vel = -(np.abs((trailer2.x**2 + trailer2.y**2) - (trailer2.x_prev**2 + trailer2.y_prev**2)))**0.5
    except:
        print('whoopsies')
        trailer1.vel = 0
        trailer2.vel = 0
    
    # Kinematic model
    x_dot = vel*np.sin(tractor.yaw)
    y_dot = vel*np.cos(tractor.yaw)
    psi_tract_dot = vel*np.tan(delta) / tractor.length
    psi_trail1_dot = vel*np.sin(art_angle)/trailer1.length + vel*L1c*np.cos(art_angle)/(tractor.length*trailer1.length)*np.tan(delta)
    psi_trailer2_dot = vel*np.sin(art_angle2)/trailer2.length

    # Update Previous positions
    tractor.x_prev = tractor.x
    tractor.y_prev = tractor.y
    trailer1.x_prev = trailer1.x
    trailer1.y_prev = trailer1.y
    trailer1.yaw_prev = trailer1.yaw
    trailer2.x_prev = trailer2.x
    trailer2.y_prev = trailer2.y
    trailer2.yaw_prev = trailer2.yaw
    
    # Update Positions
    tractor.x = tractor.x + x_dot * dt
    tractor.y = tractor.y + y_dot * dt
    tractor.yaw = tractor.yaw + psi_tract_dot * dt

    trailer1.yaw = trailer1.yaw + psi_trail1_dot * dt
    trailer1.x = tractor.x - np.sin(trailer1.yaw)*(trailer1.length)
    trailer1.y = tractor.y - np.cos(trailer1.yaw)*(trailer1.length)
    art_angle = tractor.yaw - trailer1.yaw

    trailer2.yaw = trailer2.yaw + psi_trailer2_dot * dt
    trailer2.x = trailer1.x - np.sin(trailer2.yaw)*(trailer2.length)
    trailer2.y = trailer1.y - np.cos(trailer2.yaw)*(trailer2.length)
    art_angle2 = trailer1.yaw - trailer2.yaw


    return()
    
    
    
# --------------------------------------------------------
# -----------------        Main       --------------------
# --------------------------------------------------------
# Setup Simulation
controller_choice = 'sm'
maneuver_choice = 'alley'
save2file = 0
sm.error_y1 = 0

trailer1 = trailer(0, 0, 0, 0, 0, 0, 0, 0, 0)
trailer2 = trailer(0, 0, 0, 0, 0, 0, 0, 0, 0)

trailer2.width = 2.6
trailer2.length = 16.15
tractor.width = 2.6
tractor.length = 6
trailer1.width = 2.6
trailer1.length = 16.15

sim_time = 100 # seconds
dt = 0.1 # seconds
steps_max = int(np.floor(sim_time / dt)) #iterations

# Initialize variables
delta = 0 # Steered wheel angle (rad)
simtime = 0
pid.integ = 0
pid.deriv = 0
pid.error = 0
pid.error_prev = 0
delta_prev = 0
sm.s_x = 0
sm.s_y = 0
sm.s_x_prev = 0
sm.s_y_prev = 0
simulation_variables = np.zeros([steps_max, 11])
nn.prev_scen = 0

# Setup Goal Array
goal_array = np.zeros([steps_max, 1])

# Reset starting position
if maneuver_choice == 'offset':
    # For offsetback
    tractor.x = 10 # Error measured at kingpin (meters)
    tractor.y = 55 # Error measured at kingpin (meters)
    tractor.yaw = 0 # Error measured relative to y axis (rad)
    tractor.width = 2.6
    tractor.length = 6
    tractor.x_prev = tractor.x

    trailer1.x = tractor.x
    trailer1.y = 55 - 16.15
    trailer1.yaw = 0 # Error measured relative to y axis (rad)
    trailer1.yaw_prev = trailer1.yaw
    trailer1.width = 2.6
    trailer1.length = 16.15
    trailer1.x_prev = trailer1.x

    trailer2.x = tractor.x
    trailer2.y = trailer1.y - 2.6
    trailer2.yaw = trailer1.yaw # Error measured relative to y axis (rad)
    trailer2.yaw_prev = trailer1.yaw
    trailer2.width = 2.6
    trailer2.length = 16.15
    trailer2.x_prev = trailer2.x

    vel = -1 #m/s

elif maneuver_choice == 'alley':
    # For alleypark
    tractor.x = 50 # Error measured at kingpin (meters)
    tractor.y = 5 # Error measured at kingpin (meters)
    tractor.yaw = 3.14/2 # Error measured relative to y axis (rad)
    tractor.width = 2.6
    tractor.length = 6
    tractor.x_prev = tractor.x

    trailer1.x = tractor.x
    trailer1.y = tractor.y - trailer1.length
    trailer1.yaw = tractor.yaw # Error measured relative to y axis (rad)
    trailer1.yaw_prev = trailer1.yaw
    trailer1.width = 2.6
    trailer1.length = 16.15
    trailer1.x_prev = trailer1.x

    trailer2.x = tractor.x
    trailer2.y = trailer1.y - trailer2.length
    trailer2.yaw = trailer1.yaw # Error measured relative to y axis (rad)
    trailer2.yaw_prev = trailer1.yaw
    trailer2.width = 2.6
    trailer2.length = 16.15
    trailer2.x_prev = trailer2.x

    vel = -1 #m/s

    if controller_choice == 'nn':
        filename = 'nn_offset.csv'
    elif controller_choice == 'sm':
        filename = 'sm_offset.csv'


# Run simulation
for idx in range(0, steps_max):

    if controller_choice == 'nn':
        # Call NN Controller to get Steering Angle
        goal_array[idx] = nn_controller()
        delta = pid_controller(goal_array[idx], dt)
    else:
        # Call SM Controller to get Target Art. Angle
        goal_array[idx] = sm_controller(dt)
        
        # Call PID Controller to reach Target Art. Angle
        delta = pid_controller(goal_array[idx], dt)
    
    # Rate Limit Steering Command
    delta = rate_limit(delta, delta_prev, dt)

    # Iterate positions
    delta_prev = delta
    kin_model(delta, dt, vel)
    
    # Save data
    simulation_variables[idx,:] = np.array([tractor.x, tractor.y, trailer1.x, trailer1.y, tractor.yaw, trailer1.yaw, trailer2.x, trailer2.y, trailer2.yaw, delta, simtime], dtype=object)


# Display results 
## Create Plot
fig, ax = plt.subplots(1, 1, figsize=(6,6))

## Define steps required for animation
def animate(i):
    ax.cla()
    trail_pat = plt.Rectangle(xy=(simulation_variables[0,2] - 1*math.cos(simulation_variables[0,5]), simulation_variables[0,3] + 1*math.sin(simulation_variables[0,5])), height=trailer1.length, width=trailer1.width, angle=simulation_variables[0,5],color='r')
    trail_pat.rotation_point='center'
    trail_pat.angle = simulation_variables[0,5] * -180/3.14
    tract_pat = plt.Rectangle(xy=(simulation_variables[0,0] - 1*math.cos(simulation_variables[0,4]), simulation_variables[0,1] + 1*math.sin(simulation_variables[0,4])), width=tractor.width, height=tractor.length, angle = simulation_variables[0,4], color='k')
    tract_pat.rotation_point='center'
    tract_pat.angle = simulation_variables[0,4] * -180/3.14
    trailer2_pat = plt.Rectangle(xy=(simulation_variables[0,6] - 1*math.cos(simulation_variables[0,8]), simulation_variables[0,7] + 1*math.sin(simulation_variables[0,8])), width=trailer2.width, height=trailer2.length, angle = simulation_variables[0,8], color='b')
    trailer2_pat.rotation_point='center'
    trailer2_pat.angle = simulation_variables[0,8] * -180/3.14
    ax.add_patch(trail_pat)
    ax.add_patch(tract_pat)
    ax.add_patch(trailer2_pat)
    
    trail_pat = plt.Rectangle(xy=(simulation_variables[i,2] - 1*math.cos(simulation_variables[i,5]), simulation_variables[i,3] + 1*math.sin(simulation_variables[i,5])), height=trailer1.length, width=trailer1.width, angle=simulation_variables[i,5],color='r')
    trail_pat.rotation_point='center'
    trail_pat.angle = simulation_variables[i,5] * -180/3.14
    tract_pat = plt.Rectangle(xy=(simulation_variables[i,0] - 1*math.cos(simulation_variables[i,4]), simulation_variables[i,1] + 1*math.sin(simulation_variables[i,4])), width=tractor.width, height=tractor.length, angle = simulation_variables[i,4], color='k')
    tract_pat.rotation_point='center'
    tract_pat.angle = simulation_variables[i,4] * -180/3.14
    trailer2_pat = plt.Rectangle(xy=(simulation_variables[i,6] - 1*math.cos(simulation_variables[i,8]), simulation_variables[i,7] + 1*math.sin(simulation_variables[i,8])), width=trailer2.width, height=trailer2.length, angle = simulation_variables[0,8], color='b')
    trailer2_pat.rotation_point='center'
    trailer2_pat.angle = simulation_variables[i,8] * -180/3.14
    goal_pat = plt.Rectangle(xy=(-1.8, -12), width = 3.6, height = 12, color='g')
    ax.add_patch(goal_pat)
    ax.add_patch(trail_pat)
    ax.add_patch(tract_pat)
    ax.add_patch(trailer2_pat)
    plt.plot(simulation_variables[:i,2], simulation_variables[:i,3], color='r')
    plt.plot(simulation_variables[:i,0], simulation_variables[:i,1], color='k')
    plt.plot(simulation_variables[:i,6], simulation_variables[:i,7], color='b')
    ax.set_xlim([-70, 70])
    ax.set_ylim([-70, 70])
anim = animation.FuncAnimation(fig, animate, frames=steps_max, interval=1, blit=False)
#plt.xlim([-70,70])
#plt.ylim([-10,70])
plt.plot(simulation_variables[:,7], simulation_variables[:,4]-simulation_variables[:,5],label="Observed Angle")
plt.plot(simulation_variables[:,7], simulation_variables[:,7]-simulation_variables[:,7]+1.74,'r',label="Jackknife Boundary")
plt.plot(simulation_variables[:,7], simulation_variables[:,7]-simulation_variables[:,7]-1.74,'r')
plt.title('Offset Back Articulation Angle')
plt.xlabel('Simulation Time (sec)')
plt.ylabel('Articulation Angle (rad)')
plt.legend(loc="upper right")
plt.show()