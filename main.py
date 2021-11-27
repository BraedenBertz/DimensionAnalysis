import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dt = .001
numGen = 1000
differences = 0
pl = 0
epsilon_1 = 0.0


class Brownian:
    """
    A Brownian motion class constructor
    """

    def __init__(self, x0=0):
        """
        Init class
        """
        assert (type(x0) == float or type(x0) == int or x0 is None), "Expect a float or None for the initial value"

        self.x0 = float(x0)

    def gen_random_walk(self, n_step=100):
        """
        Generate motion by random walk

        Arguments:
            n_step: Number of steps

        Returns:
            A NumPy array with `n_steps` points
        """
        # Warning about the small number of steps
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")

        w = np.ones(n_step) * self.x0

        for i in range(1, n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1, -1])
            # Weiner process
            w[i] = w[i - 1] + (yi / np.sqrt(n_step))
        return w

    def gen_normal(self, n_step=100):
        """
        Generate motion by drawing from the Normal distribution

        Arguments:
            n_step: Number of steps

        Returns:
            A NumPy array with `n_steps` points
        """
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")

        w = np.ones(n_step) * self.x0

        for i in range(1, n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i - 1] + (yi / np.sqrt(n_step))

        return w

    def stock_price(
            self,
            s0=100,
            mu=.1,
            sigma=.5,
            deltaT=52,
            dt=0.1
    ):
        """
        Models a stock price S(t) using the Weiner process W(t) as
        `S(t) = S(0).exp{(mu-(sigma^2/2).t)+sigma.W(t)}`

        Arguments:
            s0: Iniital stock price, default 100
            mu: 'Drift' of the stock (upwards or downwards), default 1
            sigma: 'Volatility' of the stock, default 1
            deltaT: The time period for which the future prices are computed, default 52 (as in 52 weeks)
            dt (optional): The granularity of the time-period, default 0.1

        Returns:
            s: A NumPy array with the simulated stock prices over the time-period deltaT
        """
        n_step = int(deltaT / dt)
        time_vector = np.linspace(0, deltaT, num=n_step)
        # Stock variation
        stock_var = (mu - (sigma ** 2 / 2)) * time_vector
        # Forcefully set the initial value to zero for the stock price simulation
        self.x0 = 0
        # Weiner process (calls the `gen_normal` method)
        weiner_process = sigma * self.gen_normal(n_step)
        # Add two time series, take exponent, and multiply by the initial stock price
        s = s0 * (np.exp(stock_var + weiner_process))

        return s


def plotGeoBrown():
    mean = 0.0
    drift = 0
    driftLastUpp = 1
    driftLastDown = 0
    # while(mean > 105 or mean < 95):

    b = Brownian()
    while True:
        drift = (driftLastUpp + driftLastDown) / 2
        print(drift)
        mean = 0
        for i in range(numGen):
            s = b.stock_price(100, drift, 1, 52, .1)
            mean += s[-1]
        mean /= numGen
        if 99 <= mean <= 101:
            return
        elif mean > 99:
            driftLastUpp = drift
            continue
        else:
            driftLastDown = drift


class Box:
    xPos = 0.0
    yPos = 0.0
    capturedData = np.ndarray
    Intersection = False
    epsilon = 0.0

    def __init__(self, xPos=0.0, yPos=0.0, capturedData=None, epsilon=0.0):
        if capturedData is None:
            capturedData = []
        self.yPos = yPos
        self.xPos = xPos
        self.capturedData = capturedData
        self.epsilon = epsilon

    def getX(self):
        return self.xPos

    def getY(self):
        return self.yPos

    def maxY(self):
        return self.yPos + self.epsilon

    def maxX(self):
        return self.xPos + self.epsilon

    def hasIntersection(self):
        for datum in self.capturedData:
            if self.yPos <= datum < self.yPos + self.epsilon:
                self.Intersection = True
                return True
        return False

    def getIntersection(self):
        return self.Intersection


def dimensionalAnalysis(epsilonDA, dimensionalData, Plot):
    # partition x and y
    xBoxescount = math.ceil(len(dimensionalData) / epsilonDA)
    yBoxescount = math.ceil((GlobalMax - abs(GlobalMin)) / epsilonDA) + 1  # max is always at least 0
    observedDataForEpsilon = math.ceil(len(dimensionalData) / xBoxescount)
    boxes = []

    print('epsilon/dt or number of observations per epsilon:' + str(observedDataForEpsilon))
    print('epsilon:' + str(epsilonDA))
    print('total number of boxes = xBoxes' + str(xBoxescount) +
          ' x yBoxes' + str(yBoxescount) + '= totalboxescount' + str(xBoxescount * yBoxescount))
    for x in range(0, xBoxescount):
        subdata = dimensionalData[x * observedDataForEpsilon:(x + 1) * observedDataForEpsilon]
        Min = min(subdata)
        Max = max(subdata)
        for y in range(math.floor((Min - abs(GlobalMin)) / epsilonDA),
                       yBoxescount - math.ceil((GlobalMax - Max) / epsilonDA) + 1):
            d = Box(x * epsilonDA, math.floor(GlobalMin / epsilonDA) * epsilonDA + y * epsilonDA, subdata, epsilonDA)
            boxes.append(d)
    for box in boxes:
        Plot.add_patch(patches.Rectangle((box.getX(), box.getY()), epsilonDA, epsilonDA, linewidth=.5, color='r',
                                         fill=box.hasIntersection(), alpha=.5))

    Plot.plot()
    print('length of boxes: ' + str(len(boxes)))
    return boxes


stocksEndValues = []
b = Brownian()
for i in range(numGen):
    s = b.stock_price(100, 0.495, 1, 52, dt)
    # s = b.gen_random_walk(int((1000)))
    # s = b.gen_normal(1000)
    stocksEndValues.append(s[-1])
    plt.plot(s)

s1 = plt.figure(1)
sns.displot(stocksEndValues, kind='kde', cut=0)
plt.show()
plt.figure(2)
axis = plt.gca()
axis.plot(s)

GlobalMax = max(s)
GlobalMin = min(s)
epsilon_1 = math.floor((GlobalMax - GlobalMin) / 10.0)
print('max: ' + str(GlobalMax))
print('min: ' + str(GlobalMin))
print('total number of datapoints: ' + str(len(s)))
epsilon_dimension_boxes = dimensionalAnalysis(epsilon_1, s, axis)
print('Printing full figure epsilon 1:')
plt.show()
plt.cla()
plt.clf()
axis = plt.gca()
axis.plot(s)
epsilon_1_dimension_boxes = dimensionalAnalysis(epsilon_1, s, axis)
plt.ylim(min(s[0:200]) - 2, max(s[0:200]) + 2)
plt.xlim(0, 200)
print('Printing 0-200 figure epsilon 1:')
plt.show()
plt.cla()
plt.clf()
axis = plt.gca()
axis.plot(s)
epsilon_2 = math.ceil(epsilon_1 / 10.0)
epsilon_2_dimension_boxes = dimensionalAnalysis(epsilon_2, s, axis)
print('Printing 0-200 figure epsilon 2:')
plt.ylim(min(s[0:200]) - 2, max(s[0:200]) + 2)
plt.xlim(0, 200)
plt.show()
N_d = 0
for boxes in epsilon_1_dimension_boxes:
    if boxes.getIntersection():
        N_d += 1
print(
    math.log2(N_d) /
    math.log2(1 / epsilon_1)
)
N_d = 0
for boxes in epsilon_2_dimension_boxes:
    if boxes.getIntersection():
        N_d += 1
print(
    math.log2(N_d) /
    math.log2(1 / epsilon_2)
)