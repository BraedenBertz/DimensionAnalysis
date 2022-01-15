import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import time
from scipy.stats import shapiro
from scipy.stats import normaltest
import seaborn as sns

dt = .01
numGen = 2
differences = 0
pl = 0
XXX = []
YYY = []
printEpsilons = False


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
        """
        # Warning about the small number of steps
        if n_step < 30:
            print("RW: WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")

        w = np.linspace(0, n_step, num=n_step, dtype=[('x', float), ('y', float)])

        for i in range(1, n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1, -1])
            # Weiner process
            w['y'][i] = w['y'][i - 1] + (yi / np.sqrt(n_step))
        return w

    def gen_normal(self, n_step=100):
        """
        Generate motion by drawing from the Normal distribution
        """
        if n_step < 30:
            print("GN: WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")

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
    """
    A Box class constructor
    """

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
        """
        determining if three points are listed in a counterclockwise order.
        So say you have three points A, B and C.
        If the slope of the line AB is less than the slope of the line AC then
        the three points are listed in a counterclockwise order.
        """
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def hasIntersection(self):
        """
        Determine if this box intersects or has data within its bounds
        """
        for a in range(len(self.capturedData) - 1):
            # see if datum is in capturedData
            if self.yPos <= self.capturedData['y'][a] <= self.yPos + self.epsilon:
                self.Intersection = True
                return True
            else:
                # Think of two line segments AB, and CD. These intersect if and only if
                # points A and B are separated by segment CD and points C and D are
                # separated by segment AB. If points A and B are separated by segment CD then
                # ACD and BCD should have opposite orientation meaning
                # either ACD or BCD is counterclockwise but not both
                # CD is always the line segment of our datapoints, but AB changes to
                # Be the line segment of the rectangle

                # A is (LL) Lower left of rectangle
                # B is (LR) Lower right of rectangle
                # C is datapoint at i
                # D is datapoint at i+1
                A = [self.xPos, self.yPos]
                B = [self.xPos + self.epsilon, self.yPos]
                C = [self.capturedData['x'][a], self.capturedData['y'][a]]
                D = [self.capturedData['x'][a + 1], self.capturedData['y'][a + 1]]
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

                # A is the Lower Left of rect
                # B is the TL
                # An intersection on the fourth line segment of the rectangle without it htitting any other line segement
                # and no data points being within the box is impossible
                # A = [self.xPos, self.yPos]
                # B = [self.xPos, self.yPos + self.epsilon]
                # if self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D):
                #     self.Intersection = True
                #     return True
        return False

    def getIntersection(self):
        return self.Intersection


def dimensionalAnalysis(epsilonDA, dimensionalData, boxes):
    """
    Create boxes that minimally cover the data given an epsilon width for the boxes
    Record those boxes to later be evaluated for intersections and be placed on graphs
    """
    observedDataForEpsilon = math.ceil(epsilonDA / dt)
    xBoxesCount = math.ceil(len(dimensionalData['y']) / observedDataForEpsilon)
    yBoxesCount = math.ceil((GlobalMax - abs(GlobalMin)) / epsilonDA) + 1  # max is always at least 0
    upper_index = 0
    for x in range(xBoxesCount):
        lower_index = max(0, upper_index - math.ceil((x * epsilonDA) % 1) - 1)
        upper_index = min(len(dimensionalData['y']), math.ceil((x + 1) * epsilonDA / dt) + 1)
        SubData = dimensionalData[lower_index: upper_index]
        Min = min(SubData['y'])
        Max = max(SubData['y'])
        for y in range(math.floor(((Min - GlobalMin) / epsilonDA)),
                       yBoxesCount - math.ceil((GlobalMax - Max) / epsilonDA) + 1):
            B = Box(x * epsilonDA, GlobalMin + y * epsilonDA,
                    SubData, epsilonDA)
            boxes.append(B)
    for B in boxes:
        B.hasIntersection()
    return boxes


def roughanddirty_dimensionalAnalysis(epsilonDA, dimensionalData):
    """
    calculate without boxes (so no graphing capability) the amount of boxes needed to cover the time series
    """
    observedDataForEpsilon = math.ceil(epsilonDA / dt)
    xBoxesCount = math.ceil(len(dimensionalData['y']) / observedDataForEpsilon)
    yBoxesCount = math.ceil((GlobalMax - abs(GlobalMin)) / epsilonDA) + 1  # max is always at least 0
    upper_index = 0
    num_boxes=0
    for x in range(xBoxesCount):
        lower_index = max(0, upper_index - math.ceil((x * epsilonDA) % 1) - 1)
        upper_index = min(len(dimensionalData['y']), math.ceil((x + 1) * epsilonDA / dt) + 1)
        SubData = dimensionalData[lower_index: upper_index]
        Min = min(SubData['y'])
        Max = max(SubData['y'])
        num_boxes = num_boxes + \
                    yBoxesCount - math.ceil((GlobalMax - Max) / epsilonDA) - \
                    math.floor(((Min - GlobalMin) / epsilonDA))

        # for y in range(math.floor(((Min - GlobalMin) / epsilonDA)),
        #                yBoxesCount - math.ceil((GlobalMax - Max) / epsilonDA)):
        #     num_boxes = num_boxes + 1

    return num_boxes


def analyseBoxes(epsilon_boxes, epsilon):
    """
    Record the number of boxes that have an intersection N_d. Then, display log N_d, log 1/epsilon
    rounded to 2 decimal digits
    """
    N_d = 0
    for ebox in epsilon_boxes:
        if ebox.getIntersection():
            N_d += 1
    x = 0
    XXX.append(math.log(1 / epsilon))
    YYY.append(math.log(N_d))
    print('{:.1e}'.format(epsilon) + '     {:.1e}'.format(epsilon) +
          '       ' + str(round(math.log(1 / epsilon), 2)) +
          '        ' + str(round(math.log(N_d), 2)))


def plotBoxes(epsilon, axis, data, xlim, boxes):
    """
    Given the plot, and our boxes (if not empty) fill in the plot with
    boxes with fill determined by their intersection status
    This should fully cover the data provided in red boxes of width epsilon
    """
    axis.plot(data['x'], data['y'])
    if len(boxes) == 0:
        boxes = dimensionalAnalysis(epsilon, data, boxes)
    for B in boxes:
        if not B.getIntersection():
            continue
        p = patches.Rectangle((B.getX(), B.getY()), epsilon, epsilon, linewidth=0, color='r',
                              fill=True, alpha=.3)
        axis.add_patch(p)
    axis.plot()
    # print('Printing figure from ' + str(xlim[0]) + ' to ' + str(xlim[1]) + ' with epsilon:' + str(epsilon))
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(min(data['y'][int(xlim[0] / dt): int(xlim[1] / dt)]) - 1,
             max(data['y'][int(xlim[0] / dt): int(xlim[1] / dt)]) + 1)
    plt.show()
    plt.cla()
    plt.clf()
    return boxes


def prettifyGraph(epsilon):
    """
    Add x and y axis titles
    """
    plt.title("Epsilon =: "+ str(epsilon))
    plt.xlabel('Time')
    plt.ylabel('Price')


def rough_limitEpsilon(t, end, DATA):
    """
    Calculate the box counting size of the data for an epsilon and continue
    until the epsilon reaches a certain threshold using rough_dimensionalAnalysis
    EX: epsilon = 11
        end = .01
        continue analysing the data with progressively smaller epsilons until
        an epsilon of .01 or less is analyzed
    """
    if printEpsilons:
        print(' ε            N(ε)         log(1/ε)     log(N(ε)) \n-------    ----------    -----------  -----------')
    while t > end:
        t = t / 2
        # prettifyGraph(t)
        # eboxes = plotBoxes(t, plt.gca(), DATA, [0, 1], [])
        # analyseBoxes(eboxes, t)
        XXX.append(math.log(1 / t))
        N_d = roughanddirty_dimensionalAnalysis(t, DATA)
        YYY.append(math.log(N_d))
        if printEpsilons:
            print('{:.1e}'.format(t)+ '     {:.1e}'.format(t)+
                '       ' + str(round(math.log(1 / t), 2)) +
              '        ' + str(round(math.log(N_d), 2)))


def priceChange(string, Data):
    d = np.asarray((Data['x'][-1], Data['y'][-1]), dtype=[('x', float), ('y', float)])

    Data1 = np.hstack([Data, d])
    Data2 = np.insert(Data, 0, (Data['x'][0], Data['y'][0]))
    Data1['y'] = np.subtract(Data1['y'],Data2['y'])
    d = Data1
    plt.plot(d['x'], d['y'])
    plt.title(string + " price changes")
    plt.show()

    dftest = adfuller(d['y'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags used', 'number of observations used'])
    for key, value in dftest[4].items():
        dfoutput['critical value (%s)' % key] = value
    print(dfoutput)
    stat, p = shapiro(d['y'])
    print('Shapiro-Wilkes test statistics=%.3f, p=%.3f' % (stat, p))
    stat, p = normaltest(d['y'])
    print('D\'Agostino k^2 Statistics=%.3f, p=%.3f' % (stat, p))
    return d


def labelPlot():
    plt.ylabel('log(n_d)')
    plt.xlabel('log(1/e)')


def maxMin(Data):
    return max(Data['y']), min(Data['y'])


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


def linReg(stri, x, y):
    labelPlot()
    plt.plot(x, y)
    result = stats.linregress(x, y)
    plt.title(str(result.slope) + ' : ' + stri + ' Slope')
    abline(1.5, result.intercept)
    plt.show()
    print(result)


def pre_process(string):
    data = pd.read_csv(string)
    d = np.linspace(0, len(data['Date']), num=len(data['Date']), dtype=[('x', float), ('y', float)])
    d['y'] = data['Open'].to_numpy()
    return d


def limitEpsilon(t, end, DATA):
    """
    Calculate the box counting size of the data for an epsilon and continue
    until the epsilon reaches a certain threshold
    EX: epsilon = 11
        end = .01
        continue analysing the data with progressively smaller epsilons until
        an epsilon of .01 or less is analyzed
    """
    print(' ε            N(ε)         log(1/ε)     log(N(ε)) \n-------    ----------    -----------  -----------')
    while t > end:
        t = t / 1.5
        # prettifyGraph(t)
        # eboxes = plotBoxes(t, plt.gca(), DATA, [0, 1], [])
        # analyseBoxes(eboxes, t)
        analyseBoxes(dimensionalAnalysis(t, DATA, []), t)


"""
Main body
"""
stocksEndValues = []
b = Brownian()
# Create a kernel density plot of the brownian motion objects (expected mean of 100)
for i in range(numGen):
    s = b.stock_price(100, 0.5, 1, 52, dt)
    # s = b.gen_random_walk(1000)
    # s = b.gen_normal(1000)
    # stocksEndValues.append(s['y'][-1])
    # plt.plot(s['x'], s['y'])

# Show the kernel densitiy plot of the endpoints of stocks, an average of 100 should
# indicate that the stochastic process has no noticeable drift
# sns.displot(stocksEndValues, kind='kde', cut=0)
# plt.axvline(100, 0, 2)
# plt.show()

GlobalMax, GlobalMin = maxMin(s)
d = priceChange("generated stock",s)
rough_limitEpsilon(61, .001, s)
linReg("generated stock", XXX, YYY)
XXX, YYY = [],[]
rough_limitEpsilon(61, .001, d)
linReg("generated stock price changes", XXX, YYY)
XXX, YYY = [], []
# DIMENSIONAL ANALYSIS OF THE NASDAQ FROM 12/03/2021 -> 12/06/2016
# DIMENSIONAL ANALYSIS OF THE NIKKEI FROM 12/07/2021 -> 11/08/2010
dt = 1
securities = {"nasdaq", "Nikkei", "BitCoin1YearPrice1-1-2021", "GOOG-1-min-8-2-2021", "enron"}
for security in securities:
    nasdaq = pre_process(security+'.csv')
    # plt.plot(nasdaq['x'], nasdaq['y'])
    # plt.show()
    d = priceChange(security, nasdaq)
    GlobalMax, GlobalMin = maxMin(nasdaq)
    rough_limitEpsilon(2000, 1, nasdaq)
    linReg(security, XXX, YYY)
    XXX, YYY = [], []
    rough_limitEpsilon(2000, 1, d)
    linReg(security+" price changes", XXX, YYY)
    XXX, YYY = [], []

# Compare to
# limitEpsilon(121, 1, enron)
# plt.ylabel('log(n_d)')
# plt.xlabel('log(1/e)')
# plt.plot(XXX,YYY)
# slope, intercept, r_value, p_value, std_err = stats.linregress(XXX, YYY)
# plt.title(slope.slope)
# plt.show()
# print(slope)

# For a more computationally efficient tho maybe less accurate calculation, consider
# using the roughanddirty_dimensionalAnalysis which doesn't create boxes and instead just computes
# the area. Limit using rough_limit

# N_e prop to e^d
# https://econwpa.ub.uni-muenchen.de/econ-wp/em/papers/0504/0504005.pdf
# http://www.scholarpedia.org/article/Grassberger-Procaccia_algorithm
