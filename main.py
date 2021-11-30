import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
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


# class LineSegment:
#     x1=0
#     x2=0
#     y1=0
#     y2=0
#     def __init__(self,x1,x2,y1,y2):
#         self.x1 = x1
#         self.x2 = x2
#         self.y1 = y1
#         self.y2 = y2
#
#     def intersection(self, ls):
#

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
        for i in range(len(self.capturedData)-1):
            # see if datum is in capturedData
            if self.yPos <= self.capturedData['y'][i] < self.yPos + self.epsilon:
                self.Intersection = True
                return True
            else:
                # A is the LL of rect
                # B is the TL
                # C is datapoint at i
                # D is datapoint at i+1
                A = [self.xPos, self.yPos]
                B = [self.xPos, self.yPos + self.epsilon]
                C = [self.capturedData['x'][i], self.capturedData['y'][i]]
                D = [self.capturedData['x'][i + 1], self.capturedData['y'][i + 1]]
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


def dimensionalAnalysis(epsilonDA, dimensionalData, Plot, boxes):
    # partition x and y
    # dt = .001 so for an epsilon, there is epsilon/dt observations per epsilon
    # if epsilon< dt, then the captured datapoints is theoretically 0, but we need at least 2 datapoints
    # to create a line that passes through it, so we set min to 2
    observedDataForEpsilon = max(2, math.ceil(epsilonDA / dt))
    xBoxescount = math.ceil(len(s['y']) / observedDataForEpsilon)
    yBoxescount = math.ceil((GlobalMax - abs(GlobalMin)) / epsilonDA) + 1  # max is always at least 0
    # print('epsilon/dt or number of observations per epsilon:' + str(observedDataForEpsilon))
    # print('epsilon:' + str(epsilonDA))
    # print('max number of boxes = xBoxes ' + str(xBoxescount) +
    #       ' x yBoxes ' + str(yBoxescount) + '= totalboxescount ' + str(xBoxescount * yBoxescount / 1000000000))
    if len(boxes) == 0:
        for x in range(xBoxescount):
            subdata = dimensionalData[math.floor(x * observedDataForEpsilon):min(len(s['y']),
                                                                                 math.ceil(
                                                                                     (x + 1) * observedDataForEpsilon)+1)]

            Min = min(subdata['y'])
            Max = max(subdata['y'])
            for y in range(math.floor((abs(Min) - abs(GlobalMin)) / epsilonDA),
                           yBoxescount - math.ceil((GlobalMax - Max) / epsilonDA) + 1):
                B = Box(x * epsilonDA, math.floor(GlobalMin / epsilonDA) * epsilonDA + y * epsilonDA, subdata,
                        epsilonDA)
                boxes.append(B)

        # print("Finished putting boxes")
        # print('length of boxes: ' + str(len(boxes)))
        for B in boxes:
            B.hasIntersection()
            Plot.add_patch((patches.Rectangle((B.getX(), B.getY()), epsilonDA, epsilonDA, linewidth=.5, color='r',
                                              fill=B.hasIntersection(), alpha=.3)))
        Plot.plot()

        return boxes
    else:
        for B in boxes:
            p = patches.Rectangle((B.getX(), B.getY()), epsilonDA, epsilonDA, linewidth=.5, color='r',
                                  fill=B.hasIntersection(), alpha=.3)
            Plot.add_patch(p)
        return boxes


def analyseBoxes(epsilon_boxes, epsilon):
    N_d = 0
    for ebox in epsilon_boxes:
        if ebox.getIntersection():
            N_d += 1
    print(
        math.log(N_d) /
        math.log(1 / epsilon)
    )


def plotBoxes(epsilon, axis, data, xlim, boxes):
    axis.plot(data['x'], data['y'])
    booooo = dimensionalAnalysis(epsilon, data, axis, boxes)
    print('Printing figure from ' + str(xlim[0]) + ' to ' + str(xlim[1]) + ' with epsilon:' + str(epsilon))
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(min(data['y'][0: int(xlim[1] * 1000)]) - 1, max(data['y'][0: int(xlim[1] * 1000)]) + 1)
    plt.show()
    plt.cla()
    plt.clf()
    return booooo


stocksEndValues = []
b = Brownian()
for i in range(numGen):
    s = b.stock_price(100, 0.495, 1, 5, dt)
    # s = b.gen_random_walk(int((1000)))
    # s = b.gen_normal(1000)
    # stocksEndValues.append(s['y'][-1])
    # plt.plot(s['x'], s['y'])

# s1 = plt.figure(1)
# sns.displot(stocksEndValues, kind='kde', cut=0)
# plt.axvline(100, 0, 2)
# plt.show()

GlobalMax = max(s['y'])
GlobalMin = min(s['y'])
print('max: ' + str(GlobalMax))
print('min: ' + str(GlobalMin))
print('total number of datapoints: ' + str(len(s['y'])))
print('dt'+str(dt))
# epsilon = (GlobalMax - GlobalMin) / 10.0
# epsilon = .5
# eboxes = plotBoxes(epsilon, plt.gca(), s, [0, 5], [])
# eboxes = plotBoxes(epsilon, plt.gca(), s, [0, 1], eboxes)
# analyseBoxes(eboxes, epsilon)
# epsilon = .05
# eboxes = plotBoxes(epsilon, plt.gca(), s, [0, 5], [])
# eboxes = plotBoxes(epsilon, plt.gca(), s, [0, 1], eboxes)
# analyseBoxes(eboxes, epsilon)
# epsilon = .005
# eboxes = plotBoxes(epsilon, plt.gca(), s, [0, 1], [])
# eboxes = plotBoxes(epsilon, plt.gca(), s, [0, .05], eboxes)
# analyseBoxes(eboxes, epsilon)
epsilon = .0005
eboxes = plotBoxes(epsilon, plt.gca(), s, [0, 1], [])
eboxes = plotBoxes(epsilon, plt.gca(), s, [0, .05], eboxes)
analyseBoxes(eboxes, epsilon)
print(0)
