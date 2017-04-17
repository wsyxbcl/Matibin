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
    if p1[0] >= d[0]:
        if p1[0] < c[0]:
            plt.step([p0[0], p1[0]], [p1[1], p1[1]], \
                     color = 'k', linestyle = '-') 
                     #p1 is used twice in purpose
            plate_num += 1
            return 0
        p2 = np.array([p1[0], line1.get_formula(p1[0])])
        plt.step([p0[0], p1[0], p2[0]], [p1[1], p1[1], p2[1]], \
                 color = 'k', linestyle = '-') 
    else:
        if p1[0] < c[0]:
            plt.step([p0[0], p1[0]], [p1[1], p1[1]], \
                     color = 'k', linestyle = '-')
            plate_num += 1
            return 0
        p2 = np.array([p1[0], line2.get_formula(p1[0])])
        plt.step([p0[0], p1[0], p2[0]], [p1[1], p1[1], p2[1]], \
                 color = 'k', linestyle = '-')
    plate_num += 1
    return 1

print("Matibin(v0.0.1(beta), Apr 17 2017) based on Python 3.5.1")
print("A open-source program to accomplish McCabe–Thiele method")
print("Source code is available at https://github.com/wsyxbcl/Matibin")
print("Feel free to use and share,just do not blame me for any future problem")
print("by wsyxbcl(yx_chai@whu.edu.cn)")

# Experiment results & some constants(e.g. the eq line)
q = input("Enter the value of q, where q = (C_pm(t_BP - t_F))/r_m + 1\
          \nq = ")
R = input("Enter the reflux ratio R, (enter 9999 for infinite reflux, \
          where R = inf.)\nR = ")
x_w = input("Enter the Bottoms composition(x_W), IN MOLE FRACTION!!!\
            x_w = ")
x_d = input("Enter the Bottoms composition(x_W), IN MOLE FRACTION!!!\
            x_D = ")
delta_k = q / (q - 1)
delta_b = 1 / (q - 1)
# R = 4
# x_w = 0.0069
# x_d = 0.81
eq_line_y = np.loadtxt(open("./eq_EtOH_data.csv", "rb"),\
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
# print plate_num
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()