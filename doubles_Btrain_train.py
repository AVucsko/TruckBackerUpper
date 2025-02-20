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

class ga:
    def __init__(self, weights, fitness, best_weights, second_weights, chrom, gen, alltime_fit, alltime_weight, numweights, numchrom):
        self.weights = weights
        self.fitness = fitness
        self.best_weights = best_weights
        self.second_weights = second_weights
        self.chrom = chrom
        self.gen = gen
        self.alltime_fit = alltime_fit
        self.alltime_weight = alltime_weight
        self.numweights = numweights
        self.numchrom = numchrom

    def setup():
        ga.numchrom = 10
        ga.fitness = np.zeros((ga.numchrom,1))
        ga.weights = np.zeros((ga.numchrom,ga.numweights))
        ga.best_weights = np.zeros((1,ga.numweights))
        ga.second_weights = np.zeros((1,ga.numweights))
        ga.gen = 0
        ga.alltime_fit = 0
        ga.alltime_weight = np.zeros((1,ga.numweights))

    def generate_weights():
        # Generate random weights
        if ga.gen == 0:
            ga.weights = np.random.normal(0, .02, (ga.numchrom, ga.numweights))
        else:
            for i in range(0, ga.numchrom):
                    ga.weights[i,:] = ga.best_weights + np.random.normal(0, .02, (1,ga.numweights))
                    for j in range(0,ga.numweights):
                        if ga.weights[i,j] > 1:
                            ga.weights[i,j] = 1
                        elif ga.weights[i,j] < -1:
                            ga.weights[i,j] = -1
    def crossover():
        # Crossover the weights
        ga.best_weights = (ga.best_weights + ga.second_weights)/2
        return()
    def fitness_calc():
        # Calculate the fitness
        ga.fitness[ga.chrom] = 1/(1 + 5*(np.abs(tractor.x) + np.abs(dolly.x) + np.abs(trailer1.x)) + (np.abs(tractor.yaw) + np.abs(dolly.yaw) + np.abs(trailer1.yaw)))
        #ga.fitness[ga.chrom] = 1/(1 + 5*(np.abs(dolly.x)) + (np.abs(dolly.yaw)))
        return(ga.fitness[ga.chrom])
    def select():
        # Select the best weights
        cnt = np.zeros((ga.numchrom,1))
        for i in range(0,ga.numchrom):
            for j in range(0,ga.numchrom):
                if ga.fitness[i] > ga.fitness[j]:
                    cnt[i] = cnt[i] + 1
        for i in range(0,ga.numchrom):
            if cnt[i] == 8:
                ga.second_weights = ga.weights[i,:]
            elif cnt[i] == 9:
                ga.best_weights = ga.weights[i,:]
        return()

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
    #k2x, qxd = 0.1, -0.20
    #k2y, qyd = -0.4, -0.2
    k1x, qx = 0.1, -0.20
    k1y, qy = -0.4, -0.4

    #Complete
    #Best Alltime Fit:  0.033937771036568226
    #Best Generation:  2166  
    #Weights:  [ 0.70748806 -1.         -0.24833786 -0.82330745  0.59996607  0.3785572
    #-0.78290688 -0.09847948]


    # Good for offsetback
    k1x, qx = ga.weights[ga.chrom, 0], ga.weights[ga.chrom, 1]
    k2x, qxd = ga.weights[ga.chrom, 2], ga.weights[ga.chrom,6]
    k1y, qy = ga.weights[ga.chrom, 3], ga.weights[ga.chrom, 4]
    k2y, qyd = ga.weights[ga.chrom, 5], ga.weights[ga.chrom, 7]
    
    
    # Sliding surface - Lat Error
    sm.error_xd = dolly.x
    sm.error_xd_dot = (dolly.x-dolly.x_prev)/dt
    s_x_d = (sm.error_xd_dot + k2x*sm.error_xd)
    #s_x_d = (sm.error_xd_dot + k2x*sm.error_xd)
    #sm.s_x = sm.error_x1_dot + sm.error_x2_dot + k1x*(sm.error_x1 + sm.error_x2)

    # Sliding surface - Trailer Angle
    sm.error_yd = dolly.yaw
    sm.error_yd_dot = (dolly.yaw-dolly.yaw_prev)/dt
    s_y_d = (sm.error_yd_dot + k2y*sm.error_yd)
    #s_y_d = (sm.error_yd_dot + k2y*sm.error_yd)
    #sm.s_y = sm.error_y1_dot + k1y*sm.error_y1
    #sm.s_y = sm.error_y1_dot  + sm.error_y2_dot + k1y*(sm.error_y1 + sm.error_y2)

    if s_x_d < 0:
        s_dot_x_d = qxd
    else:
        s_dot_x_d = -qxd

    if s_y_d < 0:
        s_dot_y_d = qyd
    else:
        s_dot_y_d = -qyd
    
    theta_x = -s_x_d + -s_dot_x_d
    theta_y = -s_y_d + -s_dot_y_d
    theta = theta_x + theta_y
    sm.s_x_prev = sm.s_x
    sm.s_y_prev = sm.s_y

    # Limit the articulation angle we can request
    if theta > 1.57:
        theta = 1.57
    elif theta < -1.57:
        theta = -1.57

    sm.error_x1 = trailer1.x
    sm.error_x1_dot = (trailer1.x-trailer1.x_prev)/dt
    sm.s_x = (sm.error_x1_dot + k1x*sm.error_x1)

    sm.error_y1_dot = ((trailer1.yaw-theta)-(sm.error_y1))/dt
    sm.error_y1 = trailer1.yaw
    sm.s_y = (sm.error_y1_dot + k1y*sm.error_y1)

    if sm.s_x < 0:
        sm.s_dot_x = qx
    else:
        sm.s_dot_x = -qx

    if sm.s_y < 0:
        sm.s_dot_y = qy
    else:
        sm.s_dot_y = -qy

    theta_x = -sm.s_x + -sm.s_dot_x
    theta_x = 0
    theta_y = -sm.s_y + -sm.s_dot_y
        
    # Limit the articulation angle we can request
    if theta > 1.57:
        theta = 1.57
    elif theta < -1.57:
        theta = -1.57
        
        
    return(theta)  

def sm_controlle_old(dt):
    # Controller Weights

    # Please use these weights
    #k1x, qx = 0.1, -0.20
    #k1y, qy = -0.4, -0.2
    
    # Good for offsetback
    k1x, qx = ga.weights[ga.chrom, 0], ga.weights[ga.chrom, 1]
    k2x, qxd = ga.weights[ga.chrom, 2], ga.weights[ga.chrom,6]
    k1y, qy = ga.weights[ga.chrom, 3], ga.weights[ga.chrom, 4]
    k2y, qyd = ga.weights[ga.chrom, 5], ga.weights[ga.chrom, 7]
    
    
    # Sliding surface - Lat Error
    sm.error_x1 = trailer1.x
    sm.error_x1_dot = (trailer1.x-trailer1.x_prev)/dt
    sm.error_xd = dolly.x
    sm.error_xd_dot = (dolly.x-dolly.x_prev)/dt
    sm.s_x = (sm.error_x1_dot + k1x*sm.error_x1) + (sm.error_xd_dot + k2x*sm.error_xd)
    #s_x_d = (sm.error_xd_dot + k2x*sm.error_xd)
    #sm.s_x = sm.error_x1_dot + sm.error_x2_dot + k1x*(sm.error_x1 + sm.error_x2)

    # Sliding surface - Trailer Angle
    sm.error_y1 = trailer1.yaw
    sm.error_y1_dot = (trailer1.yaw-trailer1.yaw_prev)/dt
    sm.error_yd = dolly.yaw
    sm.error_yd_dot = (dolly.yaw-dolly.yaw_prev)/dt
    sm.s_y = (sm.error_y1_dot + k1y*sm.error_y1) + (sm.error_yd_dot + k2y*sm.error_yd)
    #s_y_d = (sm.error_yd_dot + k2y*sm.error_yd)
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

    #if s_x_d < 0:
    #    s_dot_x_d = qxd
    #else:
    #    s_dot_x_d = -qxd

    #if s_y_d < 0:
    #    s_dot_y_d = qyd
    #else:
    #    s_dot_y_d = -qyd
    
    theta_x = -sm.s_x + -sm.s_dot_x# + -s_x_d + -s_dot_x_d
    theta_y = -sm.s_y + -sm.s_dot_y# + -s_y_d + -s_dot_y_d
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
    nn.w = ga.weights[ga.chrom,:]

    #Complete
    #Best Alltime Fit:  0.22923277349510707
    #Weights:  [-0.16050434  0.18051051 -0.96533591  0.23874888  0.31570426 -1.
    #-0.52896247  0.75347422  0.43679287  0.73438701 -0.347128    0.66018496
    #-0.51024424  0.25566677  0.5669994  -0.64711265  0.94448799  0.49281207
    #-0.09850194  0.65096077  0.93635807  0.04509822 -0.46002495  0.72295723]
	
	# Neurons
    z1 = np.tanh((tractor.yaw-trailer1.yaw) * nn.w[0])
    z2 = np.tanh((trailer1.yaw - dolly.yaw) * nn.w[1])
    z3 = np.tanh((dolly.x) * nn.w[2])
    z4 = np.tanh((dolly.yaw) * nn.w[3])
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

    # Calculate Velocities
    try:
        #trailer1.vel = -(np.abs((trailer1.x**2 - trailer1.x_prev**2)) + np.abs((trailer1.y**2 - trailer1.y_prev**2)))**0.5
        #dolly.vel = -(np.abs((dolly.x**2 - dolly.x_prev**2)) + np.abs((dolly.y**2 - dolly.y_prev**2)))**0.5
        trailer1.vel = -(np.abs((trailer1.x**2 + trailer1.y**2) - (trailer1.x_prev**2 + trailer1.y_prev**2)))**0.5
        dolly.vel = -(np.abs((dolly.x**2 + dolly.y**2) - (dolly.x_prev**2 + dolly.y_prev**2)))**0.5
    except:
        trailer1.vel = 0
        dolly.vel = 0
    
    # Kinematic model
    x_dot = vel*np.sin(tractor.yaw)
    y_dot = vel*np.cos(tractor.yaw)
    psi_tract_dot = vel*np.tan(delta) / tractor.length
    psi_trail1_dot = vel*np.sin(art_angle)/trailer1.length + vel*L1c*np.cos(art_angle)/(tractor.length*trailer1.length)*np.tan(delta)
    psi_dolly_dot = vel*np.sin(art_angle2)/dolly.length

    # Update Previous positions
    tractor.x_prev = tractor.x
    tractor.y_prev = tractor.y
    trailer1.x_prev = trailer1.x
    trailer1.y_prev = trailer1.y
    trailer1.yaw_prev = trailer1.yaw
    dolly.x_prev = dolly.x
    dolly.y_prev = dolly.y
    dolly.yaw_prev = dolly.yaw
    
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


    return()
    
    
    
# --------------------------------------------------------
# -----------------        Main       --------------------
# --------------------------------------------------------
# Setup Simulation
controller_choice = 'sm'
maneuver_choice = 'alley'
save2file = 0
sm.error_y1 = 0

dolly.width = 2.6
dolly.length = 16.15
tractor.width = 2.6
tractor.length = 6
trailer1.width = 2.6
trailer1.length = 16.15

if controller_choice == 'nn':
    ga.numweights = 24
else:
    ga.numweights = 8

if maneuver_choice == 'offset':
    # For offsetback
    dt = 0.1 #seconds
    sim_time = 100 #seconds
    steps_max = int(np.floor(sim_time / dt)) #iterations
    tractor.x = 5 # Error measured at kingpin (meters)
    tractor.y = 55 # Error measured at kingpin (meters)
    tractor.yaw = 0 # Error measured relative to y axis (rad)

    trailer1.x = tractor.x
    trailer1.y = 55 - trailer1.length
    trailer1.yaw = 0 # Error measured relative to y axis (rad)
    trailer1.yaw_prev = trailer1.yaw

    dolly.x = tractor.x
    dolly.y = trailer1.y - 2.6
    dolly.yaw = trailer1.yaw # Error measured relative to y axis (rad)
    dolly.yaw_prev = trailer1.yaw

    vel = -1 #m/s

    if controller_choice == 'nn':
        filename = 'nn_offset.csv'
    elif controller_choice == 'sm':
        filename = 'sm_offset.csv'
        
elif maneuver_choice == 'alley':
    # For alleypark
    dt = 0.1 #seconds
    sim_time = 180 #seconds
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

    vel = -1 #m/s

    if controller_choice == 'nn':
        filename = 'nn_offset.csv'
    elif controller_choice == 'sm':
        filename = 'sm_offset.csv'

ga.setup()
nn.setup()

# Run through 10,000 generations
for gen in range(0, 5000):
    ga.gen = gen + 1
    ga.generate_weights()
    fit_prev = 0
    fit = 0
    
    
    # Iterate through each chromosome
    for i in range(0, ga.numchrom):
        ga.chrom = i
    
        # Initialize variables
        delta = 0 # Steered wheel angle (rad)
        simtime = 0
        pid.integ = 0
        pid.deriv = 0
        pid.error = 0
        pid.error_prev = 0
        delta_prev = 0
        tractor.x_prev = tractor.x
        trailer1.x_prev = trailer1.x
        dolly.x_prev = dolly.x
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
            dolly.length = 16.15

            vel = -1 #m/s

        elif maneuver_choice == 'alley':
            # For alleypark
            dt = 0.1 #seconds
            steps_max = int(np.floor(sim_time / dt)) #iterations
            tractor.x = 50 # Error measured at kingpin (meters)
            tractor.y = 5 # Error measured at kingpin (meters)
            tractor.yaw = 3.14/2 # Error measured relative to y axis (rad)
            tractor.width = 2.6
            tractor.length = 6

            trailer1.x = tractor.x
            trailer1.y = tractor.y - trailer1.length
            trailer1.yaw = tractor.yaw # Error measured relative to y axis (rad)
            trailer1.yaw_prev = trailer1.yaw
            trailer1.width = 2.6
            trailer1.length = 16.15

            dolly.x = tractor.x
            dolly.y = trailer1.y - dolly.length
            dolly.yaw = trailer1.yaw # Error measured relative to y axis (rad)
            dolly.yaw_prev = trailer1.yaw
            dolly.width = 2.6
            dolly.length = 16.15

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
            simulation_variables[idx,:] = np.array([tractor.x, tractor.y, trailer1.x, trailer1.y, tractor.yaw, trailer1.yaw, dolly.x, dolly.y, dolly.yaw, delta, simtime], dtype=object)

        # Calculate Fitness
        fit = ga.fitness_calc()
        if np.max(ga.fitness) > ga.alltime_fit:
            ga.alltime_fit = np.max(ga.fitness)
            ga.alltime_weight = ga.weights[ga.chrom,:]
            simulation_variables_best = simulation_variables
            best_gen = ga.gen
        fit_prev = fit

    #if save2file == 1:
    #    np.savetxt(f,[np.max(ga.fitness)],delimiter=",",fmt="%f")
    #time = time + dt

    ga.select()
    #ga.crossover()

    # Create the animation    
    if ga.gen % 10 == 0:
        print('Generation: ', ga.gen)
        print('Best Fit: ', np.max(ga.fitness))
        #print('Weights: ', ga.best_weights)
        print('  ')

print('Complete')
print('Best Alltime Fit: ', ga.alltime_fit)
print('Best Generation: ', best_gen)
print('Weights: ', ga.alltime_weight)

# Display results 
## Create Plot
fig, ax = plt.subplots(1, 1, figsize=(6,6))

## Define steps required for animation
def animate(i):
    ax.cla()
    trail_pat = plt.Rectangle(xy=(simulation_variables_best[0,2] - 1*math.cos(simulation_variables_best[0,5]), simulation_variables_best[0,3] + 1*math.sin(simulation_variables_best[0,5])), height=trailer1.length, width=trailer1.width, angle=simulation_variables_best[0,5],color='r')
    trail_pat.rotation_point='center'
    trail_pat.angle = simulation_variables_best[0,5] * -180/3.14
    tract_pat = plt.Rectangle(xy=(simulation_variables_best[0,0] - 1*math.cos(simulation_variables_best[0,4]), simulation_variables_best[0,1] + 1*math.sin(simulation_variables_best[0,4])), width=tractor.width, height=tractor.length, angle = simulation_variables_best[0,4], color='k')
    tract_pat.rotation_point='center'
    tract_pat.angle = simulation_variables_best[0,4] * -180/3.14
    dolly_pat = plt.Rectangle(xy=(simulation_variables_best[0,6] - 1*math.cos(simulation_variables_best[0,8]), simulation_variables_best[0,7] + 1*math.sin(simulation_variables_best[0,8])), width=dolly.width, height=dolly.length, angle = simulation_variables_best[0,8], color='b')
    dolly_pat.rotation_point='center'
    dolly_pat.angle = simulation_variables_best[0,8] * -180/3.14
    ax.add_patch(trail_pat)
    ax.add_patch(tract_pat)
    ax.add_patch(dolly_pat)
    
    trail_pat = plt.Rectangle(xy=(simulation_variables_best[i,2] - 1*math.cos(simulation_variables_best[i,5]), simulation_variables_best[i,3] + 1*math.sin(simulation_variables_best[i,5])), height=trailer1.length, width=trailer1.width, angle=simulation_variables_best[i,5],color='r')
    trail_pat.rotation_point='center'
    trail_pat.angle = simulation_variables_best[i,5] * -180/3.14
    tract_pat = plt.Rectangle(xy=(simulation_variables_best[i,0] - 1*math.cos(simulation_variables_best[i,4]), simulation_variables_best[i,1] + 1*math.sin(simulation_variables_best[i,4])), width=tractor.width, height=tractor.length, angle = simulation_variables_best[i,4], color='k')
    tract_pat.rotation_point='center'
    tract_pat.angle = simulation_variables_best[i,4] * -180/3.14
    dolly_pat = plt.Rectangle(xy=(simulation_variables_best[i,6] - 1*math.cos(simulation_variables_best[i,8]), simulation_variables_best[i,7] + 1*math.sin(simulation_variables_best[i,8])), width=dolly.width, height=dolly.length, angle = simulation_variables_best[0,8], color='b')
    dolly_pat.rotation_point='center'
    dolly_pat.angle = simulation_variables_best[i,8] * -180/3.14
    goal_pat = plt.Rectangle(xy=(-1.8, -12), width = 3.6, height = 12, color='g')
    ax.add_patch(goal_pat)
    ax.add_patch(trail_pat)
    ax.add_patch(tract_pat)
    ax.add_patch(dolly_pat)
    plt.plot(simulation_variables_best[:i,2], simulation_variables_best[:i,3], color='r')
    plt.plot(simulation_variables_best[:i,0], simulation_variables_best[:i,1], color='k')
    ax.set_xlim([-70, 70])
    ax.set_ylim([-70, 70])
anim = animation.FuncAnimation(fig, animate, frames=steps_max, interval=1, blit=False)
#plt.xlim([-70,70])
#plt.ylim([-10,70])
plt.plot(simulation_variables_best[:,7], simulation_variables_best[:,4]-simulation_variables_best[:,5],label="Observed Angle")
plt.plot(simulation_variables_best[:,7], simulation_variables_best[:,7]-simulation_variables_best[:,7]+1.74,'r',label="Jackknife Boundary")
plt.plot(simulation_variables_best[:,7], simulation_variables_best[:,7]-simulation_variables_best[:,7]-1.74,'r')
plt.title('Offset Back Articulation Angle')
plt.xlabel('Simulation Time (sec)')
plt.ylabel('Articulation Angle (rad)')
plt.legend(loc="upper right")
plt.show()