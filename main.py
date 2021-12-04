import math

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dt = .001
numGen = 10
differences = 0
pl = 0


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
        time_vector = np.linspace(0, deltaT, num=n_step, dtype=[('x', float), ('y', float)])
        # Stock variation
        time_vector['y'] = time_vector['y'] * (mu - (sigma ** 2 / 2))
        # Forcefully set the initial value to zero for the stock price simulation
        self.x0 = 0
        # Weiner process (calls the `gen_normal` method)
        weiner_process = sigma * self.gen_normal(n_step)
        # Add two time series, take exponent, and multiply by the initial stock price
        time_vector['y'] = s0 * (np.exp(time_vector['y'] + weiner_process))

        return time_vector

    
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

    def ccw(self, a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    # Create a line segment from the first data point to the next one and
    # then see if that line segment intersects with the box
    def hasIntersection(self):
        for a in range(len(self.capturedData) - 1):
            # see if datum is in capturedData
            if self.yPos <= self.capturedData['y'][a] <= self.yPos + self.epsilon:
                self.Intersection = True
                return True
            else:
                # A is the LL of rect
                # B is the TL
                # C is datapoint at i
                # D is datapoint at i+1
                A = [self.xPos, self.yPos]
                B = [self.xPos, self.yPos + self.epsilon]
                C = [self.capturedData['x'][a], self.capturedData['y'][a]]
                D = [self.capturedData['x'][a + 1], self.capturedData['y'][a + 1]]
                if self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D):
                    self.Intersection = True
                    return True
                # A is LL
                # B is LR
                A = [self.xPos, self.yPos]
                B = [self.xPos + self.epsilon, self.yPos]
                if self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D):
                    self.Intersection = True
                    return True
                # A is TL
                # B is TR
                A = [self.xPos, self.yPos + self.epsilon]
                B = [self.xPos + self.epsilon, self.yPos + self.epsilon]
                if self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D):
                    self.Intersection = True
                    return True
                # A is TR
                # B is LR
                A = [self.xPos + self.epsilon, self.yPos + self.epsilon]
                B = [self.xPos + self.epsilon, self.yPos]
                if self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D):
                    self.Intersection = True
                    return True
        return False

    def getIntersection(self):
        return self.Intersection


def dimensionalAnalysis(epsilonDA, dimensionalData, boxes):
    # partition x and y
    # for an epsilon, there is epsilon/dt observations per epsilon
    # if epsilon<= dt, then the captured datapoints is theoretically 0, but we
    # still caputre data from the interpolation, thus we need 2 datapoints
    # one in front and one behind
    if epsilonDA <= dt:
        try:
            xBoxesCount = math.ceil(len(dimensionalData['y']) / epsilonDA)
            yBoxesCount = math.ceil((GlobalMax - abs(GlobalMin)) / epsilonDA) + 1  # max is always at least 0
            for x in range(xBoxesCount):
                SubData = dimensionalData[
                          math.floor(x * epsilonDA):min(len(dimensionalData), math.ceil((x + 1) * epsilonDA) + 1)]
                Min = min(SubData['y'])
                Max = max(SubData['y'])
                for y in range(math.floor((abs(Min) - abs(GlobalMin)) / epsilonDA),
                               yBoxesCount - math.ceil((GlobalMax - Max) / epsilonDA) + 1):
                    B = Box(x * epsilonDA, math.floor(GlobalMin / epsilonDA) * epsilonDA + y * epsilonDA, SubData,
                            epsilonDA)
                    boxes.append(B)
        except Exception as e:
            print(e)
    else:
        observedDataForEpsilon = math.ceil(epsilonDA / dt)
        xBoxesCount = math.ceil(len(dimensionalData['y']) / observedDataForEpsilon)
        yBoxesCount = math.ceil((GlobalMax - abs(GlobalMin)) / epsilonDA) + 1  # max is always at least 0
        for x in range(xBoxesCount):
            lower_index = math.floor(x * observedDataForEpsilon)
            upper_index = min(len(dimensionalData['y']), math.ceil((x + 1) * observedDataForEpsilon) + 1)
            SubData = dimensionalData[lower_index: upper_index]

            Min = min(SubData['y'])
            Max = max(SubData['y'])
            for y in range(math.floor((abs(Min) - abs(GlobalMin)) / epsilonDA),
                                       yBoxesCount - math.ceil((GlobalMax - Max) / epsilonDA) + 1):
                B = Box(x * epsilonDA, math.floor(GlobalMin / epsilonDA) * epsilonDA + y * epsilonDA,
                        SubData, epsilonDA)
                boxes.append(B)
    for B in boxes:
        B.hasIntersection()
    return boxes


def analyseBoxes(epsilon_boxes, epsilon):
    N_d = 0
    for ebox in epsilon_boxes:
        if ebox.getIntersection():
            N_d += 1
    print(str(epsilon)+'       '+str(N_d)+
          '        '+str(round(math.log(1/epsilon),2))+
          '        '+str(round(math.log(N_d),2)))


def plotBoxes(epsilon, axis, data, xlim, boxes):
    axis.plot(data['x'], data['y'])
    if len(boxes) == 0:
        boxes = dimensionalAnalysis(epsilon, data, boxes)
    for B in boxes:
        p = patches.Rectangle((B.getX(), B.getY()), epsilon, epsilon, linewidth=0, color='r',
                              fill=B.getIntersection(), alpha=.3)
        axis.add_patch(p)
    axis.plot()
    print('Printing figure from ' + str(xlim[0]) + ' to ' + str(xlim[1]) + ' with epsilon:' + str(epsilon))
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(min(data['y'][0: int(xlim[1] * 1/dt)]) - 1, max(data['y'][0: int(xlim[1] * 1/dt)]) + 1)
    plt.show()
    plt.cla()
    plt.clf()
    return boxes

def prettifyGraph(epsilon):
    plt.title(epsilon)
    plt.xlabel('time')
    plt.ylabel('price')

def limitEpsilon(t, end, DATA):
    while t > end:
        t = t/2
        analyseBoxes(dimensionalAnalysis(t, DATA, []), t)

stocksEndValues = []
b = Brownian()
for i in range(numGen):
    s = b.stock_price(100, 0.49, 1, 1, dt)
    # s = b.gen_random_walk(int((1000)))
    # s = b.gen_normal(1000)
    stocksEndValues.append(s['y'][-1])
    plt.plot(s['x'], s['y'])

sns.displot(stocksEndValues, kind='kde', cut=0)
plt.axvline(100, 0, 2)
plt.show()

GlobalMax = max(s['y'])
GlobalMin = min(s['y'])
print('max: ' + str(GlobalMax))
print('min: ' + str(GlobalMin))
print('total number of datapoints: ' + str(len(s['y'])))
print('dt: ' + str(dt))


print(' ε      N(ε)    log(1/ε)    log(N(ε)) \n---- ---------- ----------  -----------')
limitEpsilon(11, .009, s)

data = pd.read_csv('C:\\Users\\Braeden\\Downloads\\HistoricalData_1638573336813.csv')
data = data.iloc[::-1]
nasdaq = np.linspace(0, len(data['Date']), num=len(data['Date']), dtype=[('x', int), ('y', float)])
nasdaq['y'] = data['Open'].to_numpy()
plt.plot(nasdaq['x'], nasdaq['y'])
plt.show()
GlobalMax = max(nasdaq['y'])
GlobalMin = min(nasdaq['y'])
epsilon = 2
dt = 1
print(' ε      N(ε)    log(1/ε)    log(N(ε)) \n---- ---------- ----------  -----------')
limitEpsilon(11, .05, nasdaq)
eboxes = plotBoxes(epsilon, plt.gca(), nasdaq, [0,100], [])
plotBoxes(epsilon, plt.gca(), nasdaq, [0,10], eboxes)
eboxes = plotBoxes(epsilon/5.0, plt.gca(), nasdaq, [0,10], [])
plotBoxes(epsilon/5.0, plt.gca(), nasdaq, [0,2], eboxes)
