#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import math
import csv


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
		
class trailer1:
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

class dolly:
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

class trailer2:
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
    def __init__(self, w1, w2, w3, w4, w5, prev_scen):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.prev_scen = prev_scen
    def decide_controller():
    	# scenario == 0: Fine Controller
    	# scenario == 1: HighX Controller
    	# scenario == 2: HighYaw Controller
    	
        if nn.prev_scen == 2:
            if abs(tractor.x) <= 5 and abs(trailer1.yaw) <= 0.2:
                scenario = 0
            elif (abs(trailer1.yaw) <= 0.2) and (abs(trailer1.yaw) <= 0.2):
                scenario = 1
            else:
                scenario = 2
        if nn.prev_scen == 1:
            if abs(tractor.x) <= 5 and abs(trailer1.yaw) <= 0.2:
                scenario = 0
            elif abs(trailer1.yaw) <= 1.4:
                scenario = 1
            else:
                scenario = 2
        else:
            if abs(tractor.x) <= 5 and abs(trailer1.yaw) <= 0.2:
        	    scenario = 0
            elif abs(trailer1.yaw) <= 0.6:
                scenario = 1
            else:
                scenario = 2

        if scenario == 1:
            nn.w1, nn.w2, nn.w3, nn.w4, nn.w5 = -0.23021679, 1.08676092, -0.09843739, -3.9608538, -1.01370531
        elif scenario == 2:
            nn.w1, nn.w2, nn.w3, nn.w4, nn.w5 = -0.13244344, 0.61450203, -0.06354845, -3.70897267, -1.17183854
        else:
            nn.w1, nn.w2, nn.w3, nn.w4, nn.w5 = 1.10179206, -0.10262596, -0.30135633, 5.00553137, 4.6864815
    	
        nn.prev_scen = scenario

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
    # Controller Weights

    # Please use these weights
    #k1x, qx = 0.1, -0.20
    #k1y, qy = -0.4, -0.2
    
    # Good for offsetback
    k1x, qx = 0.12, -0.1
    k2x, q2x = 0.12, -0.1
    k1y, qy = -0.2, -0.1
    k2y, q2y = -0.2, -0.1
    
    
    # Sliding surface - Lat Error
    sm.error_x1 = trailer1.x
    sm.error_x1_dot = (trailer1.x-trailer1.x_prev)/dt
    sm.error_xd = dolly.x
    sm.error_xd_dot = (dolly.x-dolly.x_prev)/dt
    sm.error_x2 = trailer2.x
    sm.error_x2_dot = (trailer2.x-trailer2.x_prev)/dt
    #sm.s_x = (sm.error_x1_dot + k1x*sm.error_x1)
    #sm.s_x = sm.error_x1_dot + sm.error_x2_dot + k1x*(sm.error_x1 + sm.error_x2)

    # Sliding surface - Trailer Angle
    sm.error_y1 = trailer1.yaw
    sm.error_y1_dot = (trailer1.yaw-trailer1.yaw_prev)/dt
    sm.error_yd = dolly.yaw
    sm.error_yd_dot = (dolly.yaw-dolly.yaw_prev)/dt
    sm.error_y2 = trailer2.yaw
    sm.error_y2_dot = (trailer2.yaw-trailer2.yaw_prev)/dt
    #sm.s_y = sm.error_y1_dot + k1y*sm.error_y1
    #sm.s_y = sm.error_y1_dot  + sm.error_y2_dot + k1y*(sm.error_y1 + sm.error_y2)

    
    if sm.s_x < 0:
        sm.s_dot_x = qx
    else:
        sm.s_dot_x = -qx

    if sm.s_y < 0:
        sm.s_dot_y = qy
    else:
        sm.s_dot_y = -qy
    
    theta_x = -sm.s_x + -sm.s_dot_x
    theta_y = -sm.s_y + -sm.s_dot_y
    theta = theta_x + theta_y
    sm.s_x_prev = sm.s_x
    sm.s_y_prev = sm.s_y

    # If close to zero error use a simpler control scheme
    if np.abs(tractor.x) < 0.1:
        theta = trailer1.yaw
        
    # Limit the articulation angle we can request
    if theta > 1.57:
        theta = 1.57
    elif theta < -1.57:
        theta = -1.57
        
        
    return(theta)    
    
def sm_controller_single(dt):
    # Controller Weights

    # Please use these weights
    #k1x, qx = 0.1, -0.20
    #k1y, qy = -0.4, -0.2
    
    # Good for offsetback
    k1x, qx = 0.12, -0.1
    k1y, qy = -0.2, -0.1
    
    
    # Sliding surface - Lat Error
    sm.error_x = trailer1.x
    sm.error_x_dot = (trailer1.x-trailer1.x_prev)/dt
    sm.s_x = sm.error_x_dot + k1x*sm.error_x
    sm.s_dot_x = qy*(sm.s_x-sm.s_x_prev)/dt

    # Sliding surface - Trailer Angle
    sm.error_y = trailer1.yaw
    sm.error_y_dot = (trailer1.yaw-trailer1.yaw_prev)/dt
    sm.s_y = sm.error_y_dot + k1y*sm.error_y
    sm.s_dot_y = qy*(sm.s_y-sm.s_y_prev)/dt

    
    if sm.s_x < 0:
        sm.s_dot_x = qx
    else:
        sm.s_dot_x = -qx

    if sm.s_y < 0:
        sm.s_dot_y = qy
    else:
        sm.s_dot_y = -qy
    
    theta_x = -sm.s_x + -sm.s_dot_x
    theta_y = -sm.s_y + -sm.s_dot_y
    theta = theta_x + theta_y
    sm.s_x_prev = sm.s_x
    sm.s_y_prev = sm.s_y

    # If close to zero error use a simpler control scheme
    if np.abs(tractor.x) < 0.1:
        theta = trailer1.yaw
        
    # Limit the articulation angle we can request
    if theta > 1.57:
        theta = 1.57
    elif theta < -1.57:
        theta = -1.57
        
        
    return(theta)


def nn_controller():
    # Decide which controller to use
    nn.decide_controller()
	
	# Neurons
    z1 = np.tanh(tractor.yaw * nn.w1 + trailer1.yaw * nn.w2)
    z2 = np.tanh(tractor.x * nn.w3)
    z3 = np.tanh(z1*nn.w4 + z2*nn.w5)
	
    return(z3)


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
    #L_dolly = 2.6   #meters
    
    # Calculate articulation angles and apply limits
    art_angle = tractor.yaw - trailer1.yaw
    if art_angle > 1.57:
        art_angle = 1.57
        #trailer1.yaw = tractor.yaw - 1.57
    elif art_angle < -1.57:
        art_angle = -1.57
        #trailer1.yaw = tractor.yaw + 1.57

    art_angle2 = trailer1.yaw - dolly.yaw
    if art_angle2 > 1.57:
        art_angle2 = 1.57
        #dolly.yaw = trailer1.yaw - 1.57
    elif art_angle2 < -1.57:
        art_angle2 = -1.57
        #dolly.yaw = trailer1.yaw + 1.57

    art_angle3 = dolly.yaw - trailer2.yaw
    if art_angle3 > 1.57:
        art_angle3 = 1.57
        #trailer2.yaw = dolly.yaw - 1.57
    elif art_angle3 < -1.57:
        art_angle3 = -1.57
        #trailer2.yaw = dolly.yaw + 1.57
    
    # Kinematic model
    x_dot = vel*np.sin(tractor.yaw)
    y_dot = vel*np.cos(tractor.yaw)
    psi_tract_dot = vel*np.tan(delta) / tractor.length
    psi_trail1_dot = vel*np.sin(art_angle)/trailer1.length + vel*L1c*np.cos(art_angle)/(tractor.length*trailer1.length)*np.tan(delta)
    psi_dolly_dot = vel*np.sin(art_angle2)/dolly.length
    psi_trail2_dot = vel*np.sin(art_angle3)/trailer2.length

    # Update Previous positions
    tractor.x_prev = tractor.x
    tractor.y_prev = tractor.y
    trailer1.x_prev = trailer1.x
    trailer1.y_prev = trailer1.y
    trailer1.yaw_prev = trailer1.yaw
    dolly.x_prev = dolly.x
    dolly.y_prev = dolly.y
    dolly.yaw_prev = dolly.yaw
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

    dolly.yaw = dolly.yaw + psi_dolly_dot * dt
    dolly.x = trailer1.x - np.sin(dolly.yaw)*(dolly.length)
    dolly.y = trailer1.y - np.cos(dolly.yaw)*(dolly.length)
    art_angle2 = trailer1.yaw - dolly.yaw

    trailer2.yaw = trailer2.yaw + psi_trail2_dot * dt
    trailer2.x = dolly.x - np.sin(trailer2.yaw)*(trailer2.length)
    trailer2.y = dolly.y - np.cos(trailer2.yaw)*(trailer2.length)
    art_angle3 = dolly.yaw - trailer2.yaw

    return()
    
    
    
# --------------------------------------------------------
# -----------------        Main       --------------------
# --------------------------------------------------------
def main():
    # Setup Simulation
    controller_choice = 'sm'
    maneuver_choice = 'offset'
    save2file = 0
    sim_time = 100 #seconds

    if maneuver_choice == 'offset':
        # For offsetback
        dt = 0.1 #seconds
        steps_max = int(np.floor(sim_time / dt)) #iterations
        tractor.x = 5 # Error measured at kingpin (meters)
        tractor.y = 55 # Error measured at kingpin (meters)
        tractor.yaw = 0 # Error measured relative to y axis (rad)
        tractor.width = 2.6
        tractor.length = 6

        trailer1.x = tractor.x
        trailer1.y = 55 - 16.15
        trailer1.yaw = 0 # Error measured relative to y axis (rad)
        trailer1.yaw_prev = trailer1.yaw
        trailer1.width = 2.6
        trailer1.length = 16.15

        dolly.x = tractor.x
        dolly.y = trailer1.y - 2.6
        dolly.yaw = trailer1.yaw # Error measured relative to y axis (rad)
        dolly.yaw_prev = trailer1.yaw
        dolly.width = 2.6
        dolly.length = 2.6

        trailer2.x = tractor.x
        trailer2.y = dolly.y - 16.15
        trailer2.yaw = dolly.yaw # Error measured relative to y axis (rad)
        trailer2.yaw_prev = trailer1.yaw
        trailer2.width = 2.6
        trailer2.length = 16.15

        vel = -1 #m/s

        if controller_choice == 'nn':
            filename = 'nn_offset.csv'
        elif controller_choice == 'sm':
            filename = 'sm_offset.csv'
            
    elif maneuver_choice == 'alley':
        # For alleypark
        dt = 0.1 #seconds
        steps_max = int(np.floor(sim_time / dt)) #iterations
        tractor.x = 20 # Error measured at kingpin (meters)
        tractor.y = 5 # Error measured at kingpin (meters)
        tractor.yaw = 3.14/2 # Error measured relative to y axis (rad)

        trailer1.x = tractor.x
        trailer1.y = tractor.y - trailer1.length
        trailer1.yaw = tractor.yaw # Error measured relative to y axis (rad)
        trailer1.yaw_prev = trailer1.yaw

        dolly.x = tractor.x
        dolly.y = trailer1.y - dolly.length
        dolly.yaw = trailer1.yaw # Error measured relative to y axis (rad)
        dolly.yaw_prev = trailer1.yaw

        trailer2.x = tractor.x
        trailer2.y = dolly.y - trailer2.length
        trailer2.yaw = dolly.yaw # Error measured relative to y axis (rad)
        trailer2.yaw_prev = trailer1.yaw

        vel = -1 #m/s

        if controller_choice == 'nn':
            filename = 'nn_offset.csv'
        elif controller_choice == 'sm':
            filename = 'sm_offset.csv'
    
    # Initialize variables
    delta = 0 # Steered wheel angle (rad)
    time = 0
    pid.integ = 0
    pid.deriv = 0
    pid.error = 0
    pid.error_prev = 0
    delta_prev = 0
    tractor.x_prev = tractor.x
    trailer1.x_prev = trailer1.x
    dolly.x_prev = dolly.x
    trailer2.x_prev = trailer2.x
    sm.s_x = 0
    sm.s_y = 0
    sm.s_x_prev = 0
    sm.s_y_prev = 0
    simulation_variables = np.zeros([steps_max, 14])
    nn.prev_scen = 0

    # Setup Goal Array
    goal_array = np.zeros([steps_max, 1])
    
    
    # Run simulation
    with open(filename,'r') as f:
        csvreader = csv.reader(f, delimiter=",")
        for idx in range(0, steps_max):

            if controller_choice == 'nn':
                # Call NN Controller to get Steering Angle
                delta = nn_controller()
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
            simulation_variables[idx,:] = np.array([tractor.x, tractor.y, trailer1.x, trailer1.y, tractor.yaw, trailer1.yaw, dolly.x, dolly.y, dolly.yaw, trailer2.x, trailer2.y, trailer2.yaw, delta, time], dtype=object)
            if save2file == 1:
                np.savetxt(f,[simulation_variables[idx,:]],delimiter=",",fmt="%f")
            time = time + dt
    
    # Display results 
    # Create Plot
    fig, ax = plt.subplots(1, 1, figsize=(6,6))

    # Define steps required for animation
    def animate(i):
        ax.cla()
        trail_pat = plt.Rectangle(xy=(simulation_variables[0,2] - 1*math.cos(simulation_variables[0,5]), simulation_variables[0,3] + 1*math.sin(simulation_variables[0,5])), height=trailer1.length, width=trailer1.width, angle=simulation_variables[0,5],color='r')
        trail_pat.rotation_point='center'
        trail_pat.angle = simulation_variables[0,5] * -180/3.14
        tract_pat = plt.Rectangle(xy=(simulation_variables[0,0] - 1*math.cos(simulation_variables[0,4]), simulation_variables[0,1] + 1*math.sin(simulation_variables[0,4])), width=tractor.width, height=tractor.length, angle = simulation_variables[0,4], color='k')
        tract_pat.rotation_point='center'
        tract_pat.angle = simulation_variables[0,4] * -180/3.14
        dolly_pat = plt.Rectangle(xy=(simulation_variables[0,6] - 1*math.cos(simulation_variables[0,8]), simulation_variables[0,7] + 1*math.sin(simulation_variables[0,8])), width=dolly.width, height=dolly.length, angle = simulation_variables[0,8], color='k')
        dolly_pat.rotation_point='center'
        dolly_pat.angle = simulation_variables[0,8] * -180/3.14
        trail2_pat = plt.Rectangle(xy=(simulation_variables[0,9] - 1*math.cos(simulation_variables[0,11]), simulation_variables[0,10] + 1*math.sin(simulation_variables[0,11])), width=trailer2.width, height=trailer2.length,angle = simulation_variables[0,11], color='r')
        trail2_pat.rotation_point='center'
        trail2_pat.angle = simulation_variables[0,11] * -180/3.14
        ax.add_patch(trail_pat)
        ax.add_patch(tract_pat)
        ax.add_patch(dolly_pat)
        ax.add_patch(trail2_pat)
        
        trail_pat = plt.Rectangle(xy=(simulation_variables[i,2] - 1*math.cos(simulation_variables[i,5]), simulation_variables[i,3] + 1*math.sin(simulation_variables[i,5])), height=trailer1.length, width=trailer1.width, angle=simulation_variables[i,5],color='r')
        trail_pat.rotation_point='center'
        trail_pat.angle = simulation_variables[i,5] * -180/3.14
        tract_pat = plt.Rectangle(xy=(simulation_variables[i,0] - 1*math.cos(simulation_variables[i,4]), simulation_variables[i,1] + 1*math.sin(simulation_variables[i,4])), width=tractor.width, height=tractor.length, angle = simulation_variables[i,4], color='k')
        tract_pat.rotation_point='center'
        tract_pat.angle = simulation_variables[i,4] * -180/3.14
        dolly_pat = plt.Rectangle(xy=(simulation_variables[i,6] - 1*math.cos(simulation_variables[i,8]), simulation_variables[i,7] + 1*math.sin(simulation_variables[i,8])), width=dolly.width, height=dolly.length, angle = simulation_variables[0,8], color='k')
        dolly_pat.rotation_point='center'
        dolly_pat.angle = simulation_variables[i,8] * -180/3.14
        trail2_pat = plt.Rectangle(xy=(simulation_variables[i,9] - 1*math.cos(simulation_variables[i,11]), simulation_variables[i,10] + 1*math.sin(simulation_variables[i,11])), width=trailer2.width, height=trailer2.length,angle = simulation_variables[0,11], color='r')
        trail2_pat.rotation_point='center'
        trail2_pat.angle = simulation_variables[i,11] * -180/3.14
        goal_pat = plt.Rectangle(xy=(-1.8, -12), width = 3.6, height = 12, color='g')
        ax.add_patch(goal_pat)
        ax.add_patch(trail_pat)
        ax.add_patch(tract_pat)
        ax.add_patch(dolly_pat)
        ax.add_patch(trail2_pat)
        plt.plot(simulation_variables[:i,2], simulation_variables[:i,3], color='r')
        plt.plot(simulation_variables[:i,0], simulation_variables[:i,1], color='k')
        ax.set_xlim([-70, 70])
        ax.set_ylim([-70, 70])

    # Create the animation    
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
    


main()
