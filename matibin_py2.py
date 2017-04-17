#!/usr/bin/python
# -*- coding: utf-8 -*-

# The program is a accomplish of so called McCabe–Thiele method
# Just to deal with the meaningless data processing after
# the fucking Continuous distillation experiment.
# Let's call it matibin(ma for McCabe, ti for Thiele and bin for 
# respect to my roomate God Bin)
# by Yunxuan Chai(yx_chai@whu.edu.cn) 2017.4.17

import numpy as np
import matplotlib.pyplot as plt
import os
current_dir = os.getcwd()
# Initialization
x_axis = np.arange(0, 1, 0.001)


class Line:
    def __init__(self, k, b):
        """
        y = kx + b
        """
        self.k = k;
        self.b = b;
        
    def get_formula(self, x):
        return self.k * x + self.b

        
def draw(formula, y = None):
    """
    x belongs to [x_start, x_end)
    """
    global x_axis
    if y is None: # means formula exists
        y = formula(x_axis)
    plt.plot(x_axis, y, color = 'k')
    
def drop(p):
    plt.plot(p[0], p[1], 'ko')

def crossover(line1, line2):
    p = np.linalg.solve(np.array([[line1.k, -1], [line2.k, -1]]),\
                        np.array([-line1.b, -line2.b]))
    return p

def draw_stair(p0, curve_y, line1, line2):
    """
    Draw plates between given curve_y and line1 & 2
    A bug still unfixed: the corner is not a corner actually
    """
    global p2
    global x_axis
    global d
    global plate_num
    f = p0[1] * np.ones(1000)
    g = curve_y
    cross_array = np.pad(np.diff(np.array(f > g).astype(int)), \
        (1,0), 'constant', constant_values = (0,))
    cross_index = np.where(cross_array != 0)[0][0]
    p1 = np.array([x_axis[cross_index], curve_y[cross_index]])
    if p1[0] < c[0]:
        plt.step([p0[0], p1[0]], [p1[1], p1[1]], \
                 color='k', linestyle='-', linewidth=1) 
                 #p1 is used twice in purpose
        plate_num += 1
        return 0
    elif p1[0] >= d[0]:
        p2 = np.array([p1[0], line1.get_formula(p1[0])])
        plt.step([p0[0], p1[0], p2[0]], [p1[1], p1[1], p2[1]], \
                 color='k', linestyle='-',linewidth=1) 
    else:
        p2 = np.array([p1[0], line2.get_formula(p1[0])])
        plt.step([p0[0], p1[0], p2[0]], [p1[1], p1[1], p2[1]], \
                 color='k', linestyle='-', linewidth=1)
    plate_num += 1
    return 1

print "Matibin(v0.0.1(beta), Apr 17 2017) based on Python 3.5.1"
print "A open-source program to accomplish McCabe–Thiele method. See https://github.com/wsyxbcl/Matibin"
print "Feel free to use and share,just do not blame me for any future problem"
print "by wsyxbcl(yx_chai@whu.edu.cn)" 

# Experiment results & some constants(e.g. the eq line)
filename = raw_input("Enter a filename: ")
R = float(raw_input("Enter the reflux ratio R(enter inf for infinite reflux(R = inf))\nR = "))
if R > 99999:
    R = 99999
    q = 1.01 # just a random num, cause it's not important in such case
else:
    q = float(raw_input("Enter the value of q, where q = (C_pm(t_BP - t_F))/r_m + 1\nq = "))
x_w = float(raw_input("Enter the Bottoms composition(x_W), IN MOLE FRACTION!!!\nx_W = "))
x_d = float(raw_input("Enter the Distillate composition(x_W), IN MOLE FRACTION!!!\nx_D = "))
x_f = float(raw_input("Enter the Feed composition(x_W), IN MOLE FRACTION!!!\nx_F = "))
delta_k = q / (q - 1)
delta_b = x_f / (q - 1)
eq_line_y = np.loadtxt(open("./data/eq_EtOH_data.csv", "rb"),\
                       delimiter=",", skiprows= 0)
# Function of the eq line(if there is such function)
# def eq_line(x):
    # """
    # Function of the operating line
    # """
    # return 2.46 * x/(1 + 1.46 * x)
    
rec_line_k = (1.0 * R) / (R + 1)
rec_line_b = x_d / (R + 1)
rec_line = Line(rec_line_k, rec_line_b)
q_line = Line(delta_k, delta_b)
d = crossover(rec_line, q_line)
# drop(d)
c = np.array([x_w, x_w])
# drop(c)
# draw(q_line.get_formula)
str_line_k = (d[1] - c[1])/(d[0] - c[0])
str_line_b = d[1] - str_line_k * d[0]
str_line = Line(str_line_k, str_line_b)
ref_line = Line(1, 0)
# eq_line_y = eq_line(np.arange(0, 1, 0.001))

draw(None, eq_line_y)
draw(rec_line.get_formula)
draw(str_line.get_formula)
draw(ref_line.get_formula)

# Main logic
plate_num = 0
p2 = crossover(rec_line, ref_line)
for i in range(20):
    if(draw_stair(p2, eq_line_y, rec_line, str_line) == 0):
        break
        
plate_num -= 1
print "Theoretical number of plates = %d"%plate_num
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x(mole fraction of EtOH in liquid phase)')
plt.ylabel('y(mole fraction of EtOH in vapor phase)')
plt.savefig(filename+'.png', figsize=(8, 10), dpi = 400, bbox_inches='tight')
print 'The result('+filename+')is saved in'+current_dir
plt.show()
exit = raw_input("Press any key to quit.")