#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

# The program is a accomplish of so called McCabe–Thiele method
# Just to deal with the meaningless data processing after
# the fucking Continuous distillation experiment.
# Let's call it matibin(ma for McCabe, ti for Thiele and bin for 
# respect to my roomate God Bin)
# Time wasted on this: 1.5 hours(half hour for package stuff)
# by Yunxuan Chai(yx_chai@whu.edu.cn) 2017.4.17


class Line:
    def __init__(self, k, b):
        """
        y = kx + b
        """
        self.k = k;
        self.b = b;
        
    def get_formula(self, x):
        return self.k * x + self.b

        
class InputError(Exception):
    pass

class AccuracyError(InputError):
    pass
    
def draw(formula, y = None):
    """
    x belongs to [x_start, x_end)
    """
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
    global plate_num
    f = p0[1] * np.ones(10000)
    g = curve_y
    # TODO
    # This solution is less elegant, maybe there's better way
    cross_array = np.pad(np.diff(np.array(f > g).astype(int)), \
        (1,0), 'constant', constant_values = (0,))
    cross_index = np.where(cross_array != 0)[0][0]
    p1 = np.array([x_axis[cross_index], curve_y[cross_index]])
    if p1[0] > p0[0]:
        # TODO
        # Maybe fix it later. 2017.4.24(yx_chai)
        # This aims not to be triggered, just in case of an infinite loop.
        # Think I figure it out: when x_d is large, rec_line crossover eq_line first,
        # so there will be no p1 at all.
        print "Error. Something wrong with your x_D, check the unit and make sure the calculation is right"
        exit = raw_input("Type 'q' to quit.")
        raise SystemExit('Error happens in line 63, that p1[0] < p0[0].')
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
    
# Initialization
current_dir = os.getcwd()
x_axis = np.arange(0, 1, 0.0001)
print "Matibin(v0.0.2(beta), Apr 17 2017) based on Python 2.7"
print "A open-source program to accomplish McCabe_Thiele method. See https://github.com/wsyxbcl/Matibin"
print "Feel free to use and share, just do not blame me for any future problem"
print "by wsyxbcl(yx_chai@whu.edu.cn)" 

# Experimental results & some constants(e.g. the eq line)
filename = raw_input("Enter a filename: ")
while True:
    try:
        eq_line_y = np.loadtxt(open("./data/eq_EtOH_data.csv", "rb"),\
                           delimiter=",", skiprows= 0)   
        R = float(raw_input("Enter the reflux ratio R(enter inf for infinite reflux(e.g. R = 4 or R = inf))\nR = "))
        if R > 99999:
            R = 99999
            q = 1.01 # just a random num, cause it's not important in such case
        else:
            q = float(raw_input("Enter the value of q, where q = (C_pm(t_BP - t_F))/r_m + 1\nq = "))
        x_w = float(raw_input("Enter the Bottoms composition(x_W), IN MOLE FRACTION!!!(e.g. x_W = 0.007)\nx_W = "))
        if x_w <= 0.0001:
            raise AccuracyError
        x_d = float(raw_input("Enter the Distillate composition(x_W), IN MOLE FRACTION!!!\nx_D = "))
        x_f = float(raw_input("Enter the Feed composition(x_W), IN MOLE FRACTION!!!\nx_F = "))         
        rec_line_k = (1.0 * R) / (R + 1)
        rec_line_b = x_d / (R + 1)
        rec_line = Line(rec_line_k, rec_line_b)

        delta_k = q / (q - 1)
        delta_b = - x_f / (q - 1) # Bug fixed(by yx_chai 2017.4.19)
        q_line = Line(delta_k, delta_b)
        d = crossover(rec_line, q_line)
        # drop(d)
        c = np.array([x_w, x_w])
        # drop(c)
        if d[0] <= c[0]:
            raise InputError
        break        
    except ValueError as e:
        print "Value error: {0}".format(e)
        print "ValueError occurs. Make sure that you enter pure numbers not strings."
        print ""
        print "Try again here"
    except IOError as e:
        print "IOError: {0}".format(e)
        print "Make sure that you have eq_EtOH_data_10000.csv in the data directory."
        exit = raw_input("Enter 'q' to quit.")
        raise
    except InputError:
        print "Error!"
        print "Please check your input values(pay attention to the unit and the meaning of x_W & x_D)."
        print ""
        print "Try again here"
    except AccuracyError:
        print "x_w is unreasonably small, which should be at least 0.0001 in this program"
        print ""
    except Exception as e:
        print "Unexpected error: {0}".format(e)
        print "First, make sure you have a correct input."
        print "If you are sure that the input is correct. Reach me at yx_chai@whu.edu.cn."
        exit = raw_input("Enter 'q' to quit.")  
        raise
               
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
while(draw_stair(p2, eq_line_y, rec_line, str_line)):
    pass
plate_num -= 1
print "Theoretical number of plates = %d"%plate_num
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x(mole fraction of EtOH in liquid phase)')
plt.ylabel('y(mole fraction of EtOH in vapor phase)')
plt.savefig(filename+'.png', figsize=(8, 40), dpi = 400, bbox_inches='tight')
print 'The result('+filename+'.png) is saved in '+current_dir
plt.show()
exit = raw_input("Type 'q' to quit.")