
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import csv
import re
import pandas as pd
import seaborn as sns


from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

xx= []
yy = []
with open('2dData.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        xx.append(float(row[0]))
        yy.append(float(row[1]))


area = np.pi*3
color = []
with open('../dqn/akhari/actions.csv') as csv_file:
    for line in csv_file:
        color.append((float(line)))

rewards= []
with open('../dqn/akhari/rewards.csv') as csv_file:
    for line in csv_file:
        rewards.append((float(line)))
if 1:

    fig, ax = plt.subplots()
    ax.scatter(xx, yy, c=color,s=area, alpha=1)

    ##### Left - Green
    n = 300
    xy = [xx[n],yy[n]]
    #print(xy)
    # Annotate 1
    fn = '../dqn/akhari/filter_{}.png'.format(n)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.17)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                        xybox=(-0.05, 0.02),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        box_alignment=(0., 0.5),
                        arrowprops=dict(arrowstyle="<-"))
    ax.add_artist(ab)
    an1 = ax.annotate(color[n], xy=(-19.7, 0), xycoords="data",va="center", ha="center",bbox=dict(boxstyle="round", fc="w"))

    # ##### Right, Navy
    n = 50
    xy = [xx[n],yy[n]]
    # Annotate 2
    fn = '../dqn/akhari/filter_{}.png'.format(n)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.17)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                        xybox=(0.85, 0.05),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        box_alignment=(0., 0.5),
                        arrowprops=dict(arrowstyle="<-"))
    ax.add_artist(ab)
    an1 = ax.annotate(color[n], xy=(16.1, 0), xycoords="data",va="center", ha="center",bbox=dict(boxstyle="round", fc="w"))



    # ##### Bottom, Yellow
    n = 102
    xy = [xx[n],yy[n]]
    # Annotate 3
    fn = '../dqn/akhari/filter_{}.png'.format(n)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.15)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                        xybox=(0.25, -20),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        box_alignment=(0., 0.5),
                        arrowprops=dict(arrowstyle="<-"))
    ax.add_artist(ab)
    an1 = ax.annotate(color[n], xy=(-10,-14), xycoords="data",va="center", ha="center",bbox=dict(boxstyle="round", fc="w"))

    #
    # ##### Right, Navy
    n = 450
    xy = [xx[n],yy[n]]
    # Annotate 4
    fn = '../dqn/akhari/filter_{}.png'.format(n)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.15)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                        xybox=(0.85, 13),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        box_alignment=(0., 0.5),
                        arrowprops=dict(arrowstyle="<-"))
    ax.add_artist(ab)
    an1 = ax.annotate(color[n], xy=(19.7,7.2), xycoords="data",va="center", ha="center",bbox=dict(boxstyle="round", fc="w"))

    # ##### Green Top
    n = 40
    xy = [xx[n],yy[n]]
    # Annotate 4
    fn = '../dqn/akhari/filter_{}.png'.format(n)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.15)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                        xybox=(0.5, 22.15),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        box_alignment=(0., 0.5),
                        arrowprops=dict(arrowstyle="<-"))
    ax.add_artist(ab)
    an1 = ax.annotate(color[n], xy=(2.5,16), xycoords="data",va="center", ha="center",bbox=dict(boxstyle="round", fc="w"))

    # ##### Right bottom corner Blue
    n = 135
    xy = [xx[n],yy[n]]
    # Annotate 4
    fn = '../dqn/akhari/filter_{}.png'.format(n)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.15)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                        xybox=(0.85, -15),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        box_alignment=(0., 0.5),
                        arrowprops=dict(arrowstyle="<-"))
    ax.add_artist(ab)
    an1 = ax.annotate(color[n], xy=(20,-9.3), xycoords="data",va="center", ha="center",bbox=dict(boxstyle="round", fc="w"))

    # ##### Left Green 2
    n = 10
    xy = [xx[n],yy[n]]
    # Annotate 4
    fn = '../dqn/akhari/filter_{}.png'.format(n)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.15)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                        xybox=(-0.05, 15.22),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        box_alignment=(0., 0.5),
                        arrowprops=dict(arrowstyle="<-"))
    ax.add_artist(ab)
    an1 = ax.annotate(color[n], xy=(-20.4, 14), xycoords="data",va="center", ha="center",bbox=dict(boxstyle="round", fc="w"))

    # ##### Right bottom corner - BLue
    n = 232
    xy = [xx[n],yy[n]]
    # Annotate 4
    fn = '../dqn/akhari/filter_{}.png'.format(n)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.15)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                        xybox=(0.70, -19),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        box_alignment=(0., 0.5),
                        arrowprops=dict(arrowstyle="<-"))

    ax.add_artist(ab)
    an1 = ax.annotate(color[n], xy=(12, -13), xycoords="data",va="center", ha="center",bbox=dict(boxstyle="round", fc="w"))

    # ##### bottom left
    n = 241
    xy = [xx[n],yy[n]]
    # Annotate 4
    fn = '../dqn/akhari/filter_{}.png'.format(n)
    arr_img = plt.imread(fn, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.15)
    imagebox.image.axes = ax
    ab = AnnotationBbox(imagebox, xy,
                        xybox=(0.05, -19),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        box_alignment=(0., 0.5),
                        arrowprops=dict(arrowstyle="<-"))

    ax.add_artist(ab)
    an1 = ax.annotate(color[n], xy=(-20, -13), xycoords="data",va="center", ha="center",bbox=dict(boxstyle="round", fc="w"))

    ###
    # Fix the display limits to see everything
    ax.set_xlim(-25,25)
    ax.set_ylim(-20, 20)

    plt.show()
