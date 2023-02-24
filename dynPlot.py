import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set_theme(style="ticks")
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#262424'
for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '#ffffff' 

# Define the palette as a list to specify exact values
palette = sns.color_palette("rocket_r")
gridColor = '#999999'

plt.ion()
class dynPlot():

    def __init__(self, title = None, xlabel=None, ylabel=None):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], marker='o', color=palette[1],linewidth=2)
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        #Other stuff
        self.ax.grid(color=gridColor)

        self.xdata = []
        self.ydata = []

        if title  : self.ax.set_title(title)
        if xlabel : self.ax.set_xlabel(xlabel)
        if ylabel : self.ax.set_ylabel(ylabel)
    
    def __call__(self,x,y):

        # Add updated points
        self.xdata.append(x)
        self.ydata.append(y)
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(self.xdata)
        self.lines.set_ydata(self.ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def save(self,filename):
        self.figure.savefig(filename)

    def reset(self):
        self.xdata = []
        self.ydata = []
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(self.xdata)
        self.lines.set_ydata(self.ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

if __name__ == "__main__":
    import time
    d = dynPlot()
    for x in np.arange(0,10,0.5): 
        y = np.exp(-x**2)+10*np.exp(-(x-7)**2)
        d(x,y)
        time.sleep(0.1)

    if input("save plot ? [y/n]") == 'y':
        d.save(input("enter filename : "))