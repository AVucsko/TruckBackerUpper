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
        nn.w = np.zeros((1,103))
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
            ga.weights[0,:] = [ 0.70748806, -1, -0.24833786, -0.82330745, 0.59996607, 0.3785572, -0.78290688, -0.09847948]
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
        ga.fitness[ga.chrom] = 1/(1 + 5*(np.abs(tractor.x) + np.abs(trailer2.x) + np.abs(trailer1.x) + np.abs(trailer3.x) + np.abs(trailer4.x) + np.abs(trailer5.x))
         + (np.abs(tractor.yaw) + np.abs(trailer2.yaw) + np.abs(trailer1.yaw) + np.abs(trailer3.yaw) + np.abs(trailer4.yaw) + np.abs(trailer5.yaw)))
        #ga.fitness[ga.chrom] = 1/(1 + 5*(np.abs(trailer2.x)) + (np.abs(trailer2.yaw)))
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
    
def nn_controller():
    nn.w = ga.weights[ga.chrom,:]

    #Complete
    #Best Alltime Fit:  0.22923277349510707
    #Weights:  [-0.16050434  0.18051051 -0.96533591  0.23874888  0.31570426 -1.
    #-0.52896247  0.75347422  0.43679287  0.73438701 -0.347128    0.66018496
    #-0.51024424  0.25566677  0.5669994  -0.64711265  0.94448799  0.49281207
    #-0.09850194  0.65096077  0.93635807  0.04509822 -0.46002495  0.72295723]
	
	# Articulation Angles
    z1 = np.tanh((tractor.yaw-trailer1.yaw) * nn.w[0])
    z2 = np.tanh((trailer1.yaw-trailer2.yaw) * nn.w[1])
    z3 = np.tanh((trailer2.yaw-trailer3.yaw) * nn.w[2])
    z4 = np.tanh((trailer3.yaw-trailer4.yaw) * nn.w[3])
    z5 = np.tanh((trailer4.yaw-trailer5.yaw) * nn.w[4])

    # Last Trailer Location
    z6 = np.tanh((trailer5.x) * nn.w[5])
    z7 = np.tanh((trailer5.yaw) * nn.w[6])

    # Layer 1
    z8 = np.tanh(z1 * nn.w[7] + z2 * nn.w[8] + z3 * nn.w[9] + z4 * nn.w[10] + z6 * nn.w[11] + z7 * nn.w[12])
    z9 = np.tanh(z1 * nn.w[13] + z2 * nn.w[14] + z3 * nn.w[15] + z4 * nn.w[16] + z6 * nn.w[17] + z7 * nn.w[18])
    z10 = np.tanh(z1 * nn.w[19] + z2 * nn.w[20] + z3 * nn.w[21] + z4 * nn.w[22] + z6 * nn.w[23] + z7 * nn.w[24]) 
    z11 = np.tanh(z1 * nn.w[25] + z2 * nn.w[26] + z3 * nn.w[27] + z4 * nn.w[28] + z6 * nn.w[29] + z7 * nn.w[30])
    z12 = np.tanh(z1 * nn.w[31] + z2 * nn.w[32] + z3 * nn.w[33] + z4 * nn.w[34] + z6 * nn.w[35] + z7 * nn.w[36])
    z13 = np.tanh(z1 * nn.w[37] + z2 * nn.w[38] + z3 * nn.w[39] + z4 * nn.w[40] + z6 * nn.w[41] + z7 * nn.w[42]) 

    # Layer 2
    z14 = np.tanh(z8 * nn.w[43] + z9 * nn.w[44] + z10 * nn.w[45] + z11 * nn.w[46] + z12 * nn.w[47] + z13 * nn.w[68])
    z15 = np.tanh(z8 * nn.w[48] + z9 * nn.w[49] + z10 * nn.w[50] + z11 * nn.w[51] + z12 * nn.w[52] + z13 * nn.w[69])
    z16 = np.tanh(z8 * nn.w[53] + z9 * nn.w[54] + z10 * nn.w[55] + z11 * nn.w[56] + z12 * nn.w[57] + z13 * nn.w[70])
    z17 = np.tanh(z8 * nn.w[58] + z9 * nn.w[59] + z10 * nn.w[60] + z11 * nn.w[61] + z12 * nn.w[62] + z13 * nn.w[71])
    z18 = np.tanh(z8 * nn.w[63] + z9 * nn.w[64] + z10 * nn.w[65] + z11 * nn.w[66] + z12 * nn.w[67] + z13 * nn.w[72])

    # Layer 3
    z19 = np.tanh(z14 * nn.w[73] + z15 * nn.w[74] + z16 * nn.w[75] + z17 * nn.w[76] + z18 * nn.w[77])
    z20 = np.tanh(z14 * nn.w[78] + z15 * nn.w[79] + z16 * nn.w[80] + z17 * nn.w[81] + z18 * nn.w[82])
    z21 = np.tanh(z14 * nn.w[83] + z15 * nn.w[84] + z16 * nn.w[85] + z17 * nn.w[86] + z18 * nn.w[87])
    z22 = np.tanh(z14 * nn.w[88] + z15 * nn.w[89] + z16 * nn.w[90] + z17 * nn.w[91] + z18 * nn.w[92])
    z23 = np.tanh(z14 * nn.w[93] + z15 * nn.w[94] + z16 * nn.w[95] + z17 * nn.w[96] + z18 * nn.w[97])

    # Out
    z23 = np.tanh(z19 * nn.w[98] + z20 * nn.w[99] + z21 * nn.w[100] + z22 * nn.w[101] + z23 * nn.w[102])



    theta = z21*10 # Accidentally using z7 here worked in the past somehow
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
    art_angle3 = trailer2.yaw - trailer3.yaw
    art_angle4 = trailer3.yaw - trailer4.yaw
    art_angle5 = trailer4.yaw - trailer5.yaw

    # Calculate Velocities
    try:
        #trailer1.vel = -(np.abs((trailer1.x**2 - trailer1.x_prev**2)) + np.abs((trailer1.y**2 - trailer1.y_prev**2)))**0.5
        #trailer2.vel = -(np.abs((trailer2.x**2 - trailer2.x_prev**2)) + np.abs((trailer2.y**2 - trailer2.y_prev**2)))**0.5
        trailer1.vel = -(np.abs((trailer1.x**2 + trailer1.y**2) - (trailer1.x_prev**2 + trailer1.y_prev**2)))**0.5
        trailer2.vel = -(np.abs((trailer2.x**2 + trailer2.y**2) - (trailer2.x_prev**2 + trailer2.y_prev**2)))**0.5
    except:
        trailer1.vel = 0
        trailer2.vel = 0
    
    # Kinematic model
    x_dot = vel*np.sin(tractor.yaw)
    y_dot = vel*np.cos(tractor.yaw)
    psi_tract_dot = vel*np.tan(delta) / tractor.length
    psi_trail1_dot = vel*np.sin(art_angle)/trailer1.length + vel*L1c*np.cos(art_angle)/(tractor.length*trailer1.length)*np.tan(delta)
    psi_trailer2_dot = vel*np.sin(art_angle2)/trailer2.length
    psi_trail3_dot = vel*np.sin(art_angle3)/trailer3.length
    psi_trail4_dot = vel*np.sin(art_angle4)/trailer4.length
    psi_trail5_dot = vel*np.sin(art_angle5)/trailer5.length

    # Update Previous positions
    tractor.x_prev = tractor.x
    tractor.y_prev = tractor.y
    trailer1.x_prev = trailer1.x
    trailer1.y_prev = trailer1.y
    trailer1.yaw_prev = trailer1.yaw
    trailer2.x_prev = trailer2.x
    trailer2.y_prev = trailer2.y
    trailer2.yaw_prev = trailer2.yaw
    trailer3.x_prev = trailer3.x
    trailer3.y_prev = trailer3.y
    trailer3.yaw_prev = trailer3.yaw
    trailer4.x_prev = trailer4.x
    trailer4.y_prev = trailer4.y
    trailer4.yaw_prev = trailer4.yaw
    trailer5.x_prev = trailer5.x
    trailer5.y_prev = trailer5.y
    trailer5.yaw_prev = trailer5.yaw
    
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

    trailer3.yaw = trailer3.yaw + psi_trail3_dot * dt
    trailer3.x = trailer2.x - np.sin(trailer3.yaw)*(trailer3.length)
    trailer3.y = trailer2.y - np.cos(trailer3.yaw)*(trailer3.length)
    art_angle3 = trailer2.yaw - trailer3.yaw

    trailer4.yaw = trailer4.yaw + psi_trail4_dot * dt
    trailer4.x = trailer3.x - np.sin(trailer4.yaw)*(trailer4.length)
    trailer4.y = trailer3.y - np.cos(trailer4.yaw)*(trailer4.length)
    art_angle4 = trailer3.yaw - trailer4.yaw

    trailer5.yaw = trailer5.yaw + psi_trail5_dot * dt
    trailer5.x = trailer4.x - np.sin(trailer5.yaw)*(trailer5.length)
    trailer5.y = trailer4.y - np.cos(trailer5.yaw)*(trailer5.length)
    art_angle5 = trailer4.yaw - trailer5.yaw


    return()
    
    
    
# --------------------------------------------------------
# -----------------        Main       --------------------
# --------------------------------------------------------
# Setup Simulation
controller_choice = 'nn'
maneuver_choice = 'alley'
save2file = 0
sm.error_y1 = 0

trailer1 = trailer(0,0,0,0,0,0,0,0,0)
trailer2 = trailer(0,0,0,0,0,0,0,0,0)
trailer3 = trailer(0,0,0,0,0,0,0,0,0)
trailer4 = trailer(0,0,0,0,0,0,0,0,0)
trailer5 = trailer(0,0,0,0,0,0,0,0,0)

tractor.width = 2.6
tractor.length = 6
trailer1.width = 2.6
trailer1.length = 16.15
trailer2.width = 2.6
trailer2.length = 16.15
trailer3.width = 2.6
trailer3.length = 16.15
trailer4.width = 2.6
trailer4.length = 16.15
trailer5.width = 2.6
trailer5.length = 16.15



if controller_choice == 'nn':
    ga.numweights = 103
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

    trailer2.x = tractor.x
    trailer2.y = trailer1.y - 2.6
    trailer2.yaw = trailer1.yaw # Error measured relative to y axis (rad)
    trailer2.yaw_prev = trailer1.yaw

    vel = -1 #m/s

    if controller_choice == 'nn':
        filename = 'nn_offset.csv'
    elif controller_choice == 'sm':
        filename = 'sm_offset.csv'
        
elif maneuver_choice == 'alley':
    # For alleypark
    dt = 0.1 #seconds
    sim_time = 200 #seconds
    steps_max = int(np.floor(sim_time / dt)) #iterations
    tractor.x = 100 # Error measured at kingpin (meters)
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

    trailer2.x = tractor.x
    trailer2.y = trailer1.y - trailer2.length
    trailer2.yaw = trailer1.yaw # Error measured relative to y axis (rad)
    trailer2.yaw_prev = trailer1.yaw
    trailer2.width = 2.6
    trailer2.length = 16.15

    trailer3.x = tractor.x
    trailer3.y = trailer2.y - trailer1.length
    trailer3.yaw = tractor.yaw # Error measured relative to y axis (rad)
    trailer3.yaw_prev = trailer1.yaw
    trailer3.width = 2.6
    trailer3.length = 16.15

    trailer4.x = tractor.x
    trailer4.y = trailer3.y - trailer1.length
    trailer4.yaw = tractor.yaw # Error measured relative to y axis (rad)
    trailer4.yaw_prev = trailer1.yaw
    trailer4.width = 2.6
    trailer4.length = 16.15

    trailer5.x = tractor.x
    trailer5.y = trailer4.y - trailer1.length
    trailer5.yaw = tractor.yaw # Error measured relative to y axis (rad)
    trailer5.yaw_prev = trailer1.yaw
    trailer5.width = 2.6
    trailer5.length = 16.15

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
        trailer2.x_prev = trailer2.x
        sm.s_x = 0
        sm.s_y = 0
        sm.s_x_prev = 0
        sm.s_y_prev = 0
        simulation_variables = np.zeros([steps_max, 20])
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
            trailer1.y = 55 - trailer1.length
            trailer1.yaw = 0 # Error measured relative to y axis (rad)
            trailer1.yaw_prev = trailer1.yaw
            trailer1.width = 2.6
            trailer1.length = 16.15

            trailer2.x = tractor.x
            trailer2.y = trailer1.y - 2.6
            trailer2.yaw = trailer1.yaw # Error measured relative to y axis (rad)
            trailer2.yaw_prev = trailer1.yaw
            trailer2.width = 2.6
            trailer2.length = 16.15

            vel = -1 #m/s

        elif maneuver_choice == 'alley':
            # For alleypark
            dt = 0.1 #seconds
            steps_max = int(np.floor(sim_time / dt)) #iterations
            tractor.x = 100 # Error measured at kingpin (meters)
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

            trailer2.x = tractor.x
            trailer2.y = trailer1.y - trailer2.length
            trailer2.yaw = trailer1.yaw # Error measured relative to y axis (rad)
            trailer2.yaw_prev = trailer1.yaw
            trailer2.width = 2.6
            trailer2.length = 16.15

            trailer3.x = tractor.x
            trailer3.y = trailer2.y - trailer1.length
            trailer3.yaw = tractor.yaw # Error measured relative to y axis (rad)
            trailer3.yaw_prev = trailer1.yaw
            trailer3.width = 2.6
            trailer3.length = 16.15

            trailer4.x = tractor.x
            trailer4.y = trailer3.y - trailer1.length
            trailer4.yaw = tractor.yaw # Error measured relative to y axis (rad)
            trailer4.yaw_prev = trailer1.yaw
            trailer4.width = 2.6
            trailer4.length = 16.15

            trailer5.x = tractor.x
            trailer5.y = trailer4.y - trailer1.length
            trailer5.yaw = tractor.yaw # Error measured relative to y axis (rad)
            trailer5.yaw_prev = trailer1.yaw
            trailer5.width = 2.6
            trailer5.length = 16.15

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
            simulation_variables[idx,:] = np.array([tractor.x, tractor.y, trailer1.x, trailer1.y, tractor.yaw, trailer1.yaw, trailer2.x, trailer2.y, trailer2.yaw, trailer3.x, trailer3.y, trailer3.yaw, trailer4.x, trailer4.y, trailer4.yaw, trailer5.x, trailer5.y, trailer5.yaw, delta, simtime], dtype=object)
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
    trailer2_pat = plt.Rectangle(xy=(simulation_variables_best[0,6] - 1*math.cos(simulation_variables_best[0,8]), simulation_variables_best[0,7] + 1*math.sin(simulation_variables_best[0,8])), width=trailer2.width, height=trailer2.length, angle = simulation_variables_best[0,8], color='b')
    trailer2_pat.rotation_point='center'
    trailer2_pat.angle = simulation_variables_best[0,8] * -180/3.14
    trailer3_pat = plt.Rectangle(xy=(simulation_variables_best[0,9] - 1*math.cos(simulation_variables_best[0,11]), simulation_variables_best[0,10] + 1*math.sin(simulation_variables_best[0,11])), width=trailer2.width, height=trailer2.length, angle = simulation_variables_best[0,11], color='r')
    trailer3_pat.rotation_point='center'
    trailer3_pat.angle = simulation_variables_best[0,11] * -180/3.14
    trailer4_pat = plt.Rectangle(xy=(simulation_variables_best[0,12] - 1*math.cos(simulation_variables_best[0,14]), simulation_variables_best[0,13] + 1*math.sin(simulation_variables_best[0,14])), width=trailer2.width, height=trailer2.length, angle = simulation_variables_best[0,14], color='b')
    trailer4_pat.rotation_point='center'
    trailer4_pat.angle = simulation_variables_best[0,14] * -180/3.14
    trailer5_pat = plt.Rectangle(xy=(simulation_variables_best[0,15] - 1*math.cos(simulation_variables_best[0,17]), simulation_variables_best[0,18] + 1*math.sin(simulation_variables_best[0,17])), width=trailer2.width, height=trailer2.length, angle = simulation_variables_best[0,17], color='r')
    trailer5_pat.rotation_point='center'
    trailer5_pat.angle = simulation_variables_best[0,17] * -180/3.14
    #ax.add_patch(trail_pat)
    #ax.add_patch(tract_pat)
    #ax.add_patch(trailer2_pat)
    #ax.add_patch(trailer3_pat)
    #ax.add_patch(trailer4_pat)
    #ax.add_patch(trailer5_pat)
    
    trail_pat = plt.Rectangle(xy=(simulation_variables_best[i,2] - 1*math.cos(simulation_variables_best[i,5]), simulation_variables_best[i,3] + 1*math.sin(simulation_variables_best[i,5])), height=trailer1.length, width=trailer1.width, angle=simulation_variables_best[i,5],color='r')
    trail_pat.rotation_point='center'
    trail_pat.angle = simulation_variables_best[i,5] * -180/3.14
    tract_pat = plt.Rectangle(xy=(simulation_variables_best[i,0] - 1*math.cos(simulation_variables_best[i,4]), simulation_variables_best[i,1] + 1*math.sin(simulation_variables_best[i,4])), width=tractor.width, height=tractor.length, angle = simulation_variables_best[i,4], color='k')
    tract_pat.rotation_point='center'
    tract_pat.angle = simulation_variables_best[i,4] * -180/3.14
    trailer2_pat = plt.Rectangle(xy=(simulation_variables_best[i,6] - 1*math.cos(simulation_variables_best[i,8]), simulation_variables_best[i,7] + 1*math.sin(simulation_variables_best[i,8])), width=trailer2.width, height=trailer2.length, angle = simulation_variables_best[0,8], color='b')
    trailer2_pat.rotation_point='center'
    trailer2_pat.angle = simulation_variables_best[i,8] * -180/3.14
    goal_pat = plt.Rectangle(xy=(-1.8, -12), width = 3.6, height = 12, color='g')
    trailer3_pat = plt.Rectangle(xy=(simulation_variables_best[i,9] - 1*math.cos(simulation_variables_best[i,11]), simulation_variables_best[i,10] + 1*math.sin(simulation_variables_best[i,11])), width=trailer2.width, height=trailer2.length, angle = simulation_variables_best[i,11], color='r')
    trailer3_pat.rotation_point='center'
    trailer3_pat.angle = simulation_variables_best[i,11] * -180/3.14
    trailer4_pat = plt.Rectangle(xy=(simulation_variables_best[i,12] - 1*math.cos(simulation_variables_best[i,14]), simulation_variables_best[i,13] + 1*math.sin(simulation_variables_best[i,14])), width=trailer2.width, height=trailer2.length, angle = simulation_variables_best[i,14], color='b')
    trailer4_pat.rotation_point='center'
    trailer4_pat.angle = simulation_variables_best[i,14] * -180/3.14
    trailer5_pat = plt.Rectangle(xy=(simulation_variables_best[i,15] - 1*math.cos(simulation_variables_best[i,17]), simulation_variables_best[i,16] + 1*math.sin(simulation_variables_best[i,17])), width=trailer2.width, height=trailer2.length, angle = simulation_variables_best[i,17], color='r')
    trailer5_pat.rotation_point='center'
    trailer5_pat.angle = simulation_variables_best[i,17] * -180/3.14
    ax.add_patch(goal_pat)
    ax.add_patch(trail_pat)
    ax.add_patch(tract_pat)
    ax.add_patch(trailer2_pat)
    ax.add_patch(trailer3_pat)
    ax.add_patch(trailer4_pat) 
    ax.add_patch(trailer5_pat)
    plt.plot(simulation_variables_best[:i,2], simulation_variables_best[:i,3], color='r')
    plt.plot(simulation_variables_best[:i,0], simulation_variables_best[:i,1], color='k')
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
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