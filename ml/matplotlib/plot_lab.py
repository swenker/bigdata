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

def plot_axes_sin_cos():
    plt.figure(figsize=(4,3))
    plt.subplot(221)
    plt.grid(True)
    plt.xlabel('I am X')
    plt.ylabel('I am Y')

    X=np.array([1,2,3])
    y=X*2

    PI = np.pi
    plt.xticks([0,PI/4,2*PI/4,3*PI/4,PI,5*PI/4,6*PI/4,7*PI/4,2*PI])
    # plt.xlim(-PI,PI)
    # plt.ylim(-1,1)
    # X=[0,PI/8,PI/7,PI/6,PI/5,PI/4,PI/2,PI]
    X=np.arange(0,2*PI,0.1)
    y=np.sin(X)
    plt.plot(X,y,'rD',label='sin(x)')
    plt.plot(X-PI/2,np.cos(X),'bs',label='cos(x)')
    plt.legend(loc='upper left')
    plt.show()

def plot_axes_exponent():
    plt.figure(figsize=(4,3))

    plt.subplot(221)
    plt.grid(True)
    plt.xlabel('I am X')
    plt.ylabel('I am Y')

    X=np.arange(0,10)

    plt.plot(X,X,'r',label='y=X')
    plt.plot(X,X**2,'b',label='X2')
    plt.plot(X,X**3,'g',label='X3')
    plt.plot(X,X**4,'y',label='X4')
    plt.legend(loc='upper left')
    plt.show()

def plot_usage():
    plt.figure(figsize=(4,3))

    plt.grid(True)
    plt.xlabel('I am X')
    plt.ylabel('I am Y')

    X=np.arange(-10,10,.1)
    y = 15**2-X**2
    plt.plot(X,y,'r.',label='y=X')
    # plt.title('How plot works')
    plt.show()

# point_plot()
# plot_axes_exponent()
plot_usage()