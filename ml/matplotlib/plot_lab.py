import numpy as np
import matplotlib.pyplot as plt

def line_plot():

    plt.figure('I am figure title')

    #The last one in the grid
    plt.subplot(2,2,1)
    plt.title('power(x)')

    x=np.array([1,2,3,4])
    x=np.arange(1,4,0.1)

    # plt.plot([1,2,3,4],[2,4,6,8],'ro')
    # plt.plot([1,2,3,4],[2,4,6,8],'r.')
    # plt.plot([1,2,3,4],[2,4,6,8],'r<')
    plt.plot(x,x**2,'r')

    #This will open a new figure window
    # plt.figure(2)
    plt.subplot(222)
    plt.title('square(x)')
    plt.plot(x,x**0.5,'r')


    plt.subplot(223)
    plt.plot(x,np.cos(x),'r')

    #the range of the axis [xmin,xmax,ymin,ymax]
    plt.axis([0,5,0,10])
    plt.ylabel("I am Y")
    plt.show()


def point_plot():
    x=[1,2,3,4,5,6,7,8]
    y=np.power(x,4)

    plt.scatter(x,y)

    plt.grid(True)
    # plt.colorbar()
    # plt.colormaps()
    # plt.colors()
    plt.show()


# line_plot()
point_plot()