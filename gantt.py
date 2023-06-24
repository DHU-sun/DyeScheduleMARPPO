#!/usr/bin/env python
# pylint: disable=R0902, R0903
"""
Gantt.py is a simple class to render Gantt charts, as commonly used in
"""

# handling of TeX support:
# on Linux assume TeX
# on OSX add TexLive if present
import os
import numpy as np
import platform
from operator import sub

LATEX = True
if (platform.system() == 'Darwin') & os.path.isdir('/usr/texbin'):
    os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'
elif (platform.system() == 'Linux') & os.path.isfile('/usr/bin/latex'):
    LATEX = True
else:
    LATEX = False

# setup pyplot w/ tex support
import matplotlib.pyplot as plt
from matplotlib import rc

if LATEX:
    rc('text', usetex=True)

fontdict = {'weight': 'medium',
            'color': 'black',
            'size': 10,
            }
sample_dict = {
    "packages": [
        {"part_num": "Part20",
         "label": "WP 1-1",
         "start": 0,
         "end": 2,
         "milestones": [2],
         "legend": "worker one"
         },
        {"part_num": "Part20",
         "label": "WP 1-2",
         "start": 2,
         "end": 4,
         "milestones": [3, 4]
         },
        {"label": "WP 2-1",
         "start": 3,
         "end": 5,
         "milestones": [5],
         "part_num": "Part76",
         "legend": "worker two"
         },
        {"label": "WP 2-2",
         "start": 6,
         "end": 8,
         "milestones": [8],
         "part_num": "Part54"
         },
        {"label": "WP 2-3",
         "start": 7,
         "end": 9,
         "milestones": [8],
         "part_num": "Part20"
         },
        {"label": "WP 2-4",
         "start": 8,
         "end": 9,
         "milestones": [9],
         "part_num": "Part7"
         },
        {"label": "WP 3-1",
         "start": 2,
         "end": 6,
         "milestones": [4, 6],
         "part_num": "Part2",
         "legend": "worker three"
         },
        {"label": "WP 3-2",
         "start": 4,
         "end": 8,
         "milestones": [8],
         "part_num": "Part3"
         },
        {"label": "WP 3-3",
         "start": 6,
         "end": 12,
         "part_num": "Part89"
         }
    ],

    "title": " Sample GANTT for \\textbf{myProject}",
    "xlabel": "time (weeks)",
    "xticks": [2, 4, 6, 8, 10, 12]
}
color_dict = {
    0: '#F0F8FF',
    1: '#FAEBD7',
    2: '#00FFFF',
    3: '#7FFFD4',
    4: '#F0FFFF',
    5: '#F5F5DC',
    6: '#FFE4C4',
    7: '#000000',
    8: '#FFEBCD',
    9: '#0000FF',
    10: '#8A2BE2',
    11: '#A52A2A',
    12: '#DEB887',
    13: '#5F9EA0',
    14: '#7FFF00',
    15: '#D2691E',
    16: '#FF7F50',
    17: '#6495ED',
    18: '#FFF8DC',
    19: '#DC143C',
    20: '#00FFFF',
    21: '#00008B',
    22: '#008B8B',
    23: '#B8860B',
    24: '#A9A9A9',
    25: '#006400',
    26: '#BDB76B',
    27: '#8B008B',
    28: '#556B2F',
    29: '#FF8C00',
    30: '#9932CC',
    31: '#8B0000',
    32: '#E9967A',
    33: '#8FBC8F',
    34: '#483D8B',
    35: '#2F4F4F',
    36: '#00CED1',
    37: '#9400D3',
    38: '#FF1493',
    39: '#00BFFF',
    40: '#696969',
    41: '#1E90FF',
    42: '#B22222',
    43: '#FFFAF0',
    44: '#228B22',
    45: '#FF00FF',
    46: '#DCDCDC',
    47: '#F8F8FF',
    48: '#FFD700',
    49: '#DAA520',
    50: '#808080',
    51: '#008000',
    52: '#ADFF2F',
    53: '#F0FFF0',
    54: '#FF69B4',
    55: '#CD5C5C',
    56: '#4B0082',
    57: '#FFFFF0',
    58: '#F0E68C',
    59: '#E6E6FA',
    60: '#FFF0F5',
    61: '#7CFC00',
    62: '#FFFACD',
    63: '#ADD8E6',
    64: '#F08080',
    65: '#E0FFFF',
    66: '#FAFAD2',
    67: '#90EE90',
    68: '#D3D3D3',
    69: '#FFB6C1',
    70: '#FFA07A',
    71: '#20B2AA',
    72: '#87CEFA',
    73: '#778899',
    74: '#B0C4DE',
    75: '#FFFFE0',
    76: '#00FF00',
    77: '#32CD32',
    78: '#FAF0E6',
    79: '#FF00FF',
    80: '#800000',
    81: '#66CDAA',
    82: '#0000CD',
    83: '#BA55D3',
    84: '#9370DB',
    85: '#3CB371',
    86: '#7B68EE',
    87: '#00FA9A',
    88: '#48D1CC',
    89: '#C71585',
    90: '#191970',
    91: '#F5FFFA',
    92: '#FFE4E1',
    93: '#FFE4B5',
    94: '#FFDEAD',
    95: '#000080',
    96: '#FDF5E6',
    97: '#808000',
    98: '#6B8E23',
    99: '#FFA500',
    100: '#FF4500',
    101: '#DA70D6',
    102: '#EEE8AA',
    103: '#98FB98',
    104: '#AFEEEE',
    105: '#DB7093',
    106: '#FFEFD5',
    107: '#FFDAB9',
    108: '#CD853F',
    109: '#FFC0CB',
    110: '#DDA0DD',
    111: '#B0E0E6',
    112: '#800080',
    113: '#FF0000',
    114: '#BC8F8F',
    115: '#4169E1',
    116: '#8B4513',
    117: '#FA8072',
    118: '#FAA460',
    119: '#2E8B57',
    120: '#FFF5EE',
    121: '#A0522D',
    122: '#C0C0C0',
    123: '#87CEEB',
    124: '#6A5ACD',
    125: '#708090',
    126: '#FFFAFA',
    127: '#00FF7F',
    128: '#4682B4',
    129: '#D2B48C',
    130: '#008080',
    131: '#D8BFD8',
    132: '#FF6347',
    133: '#40E0D0',
    134: '#EE82EE',
    135: '#F5DEB3',
    136: '#FFFFFF',
    137: '#F5F5F5',
    138: '#FFFF00',
    139: '#9ACD32'}


class Package(object):
    """Encapsulation of a work package

    A work package is instantiate from a dictionary. It **has to have**
    a label, astart and an end. Optionally it may contain milestones
    and a color

    :arg str pkg: dictionary w/ package data name
    """

    def __init__(self, pkg):

        self.label = pkg['label']
        self.start = pkg['start']
        self.end = pkg['end']
        self.part_num = pkg['part_num']
        self.delay = pkg['delay']

        if self.start < 0 or self.end < 0:
            raise ValueError("Package cannot begin at t < 0")
        if self.start > self.end:
            raise ValueError("Cannot end before started")

        try:
            self.milestones = pkg['milestones']
        except KeyError:
            pass

        try:
            if self.delay > 0:
                self.color = self.get_part_color()
            else:
                self.color = 'lightgrey'
        except KeyError:
            self.color = '#9ACD32'

        try:
            self.legend = pkg['legend']
        except KeyError:
            self.legend = None

    def get_part_color(self):
        color_key = self.part_num % len(color_dict)

        return color_dict.get(color_key)


class Gantt(object):
    """Gantt
    Class to render a simple Gantt chart, with optional milestones
    """

    def __init__(self):
        """ Instantiation

        Create a new Gantt using the data in the file provided
        or the sample data that came along with the script

        :arg str dataFile: file holding Gantt data
        """

    def initialize(self, algorithm, ex, dict):
        self.ex = ex
        self.Algorithm = algorithm
        self.sampledict = dict

        # some lists needed
        self.packages = []
        self.labels = []

        self._loadData()
        self._procData()

        # assemble colors
        self.colors = []
        for pkg in self.packages:
            self.colors.append(pkg.color)

    def _loadData(self):
        """ Load data from a JSON file that has to have the keys:
            packages & title. Packages is an array of objects with
            a label, start and end property and optional milesstones
            and color specs.
        """

        # load data
        data = self.sampledict

        # must-haves
        self.max_length = data['max_length']
        self.title = data['title']
        self.figure_num = data['figure_num']

        for pkg in data['packages']:
            self.packages.append(Package(pkg))

        self.labels = [pkg['label'] for pkg in data['packages']]
        # self.labels.sort()

        # optionals
        self.milestones = {}
        self.milestones['y_label'] = []
        self.milestones['x_value'] = []

        for pkg in self.packages:
            try:
                self.milestones['x_value'].append(pkg.milestones)
                self.milestones['y_label'].append(pkg.label)

            except AttributeError:
                pass

        try:
            self.xlabel = data['xlabel']
        except KeyError:
            self.xlabel = ""

        try:
            self.ylabel = data['ylabel']
        except KeyError:
            self.ylabel = ""

        try:
            self.xticks = data['xticks']
        except KeyError:
            self.xticks = ""

        try:
            self.xticks_label = data['xticks_label']
        except KeyError:
            self.xticks_label = ""

    def _procData(self):
        """ Process data to have all values needed for plotting
        """

        # parameters for bars
        self.nPackages = len(self.labels)
        self.start = [None] * self.nPackages
        self.end = [None] * self.nPackages
        self.edge_color = [None] * self.nPackages
        i = 0

        for pkg in self.packages:
            # idx = self.labels.index(pkg.label)
            self.start[i] = pkg.start
            self.end[i] = pkg.end
            if pkg.delay > 0:
                edge_color = 'red'
            else:
                edge_color = 'black'
            self.edge_color[i] = edge_color
            i += 1

        self.durations = map(sub, self.end, self.start)
        self.yPos = self.labels

    def format(self):
        """ Format various aspect of the plot, such as labels,ticks, BBox
        :todo: Refactor to use a settings object
        """

        # format axis
        plt.tick_params(
            axis='both',  # format x and y
            which='both',  # major and minor ticks affected
            bottom='on',  # bottom edge ticks are on
            top='off',  # top, left and right edge ticks are off
            left='off',
            right='off')

        # tighten axis but give a little room from bar height

        plt.xlim(0, max(self.end))
        plt.ylim(min(self.labels) - .5, max(self.labels) + .5)

        # add title and package names
        plt.yticks(self.yPos, self.labels)
        plt.title(self.title)

        if self.xlabel:
            plt.xlabel(self.xlabel)

        if self.ylabel:
            plt.ylabel(self.ylabel)

        if self.xticks_label:
            plt.xticks(self.xticks, map(str, self.xticks_label), rotation=90)
        else:
            plt.xticks(self.xticks, map(str, self.xticks))


    def addMilestones(self):
        """Add milestones to GANTT chart.
        The milestones are simple yellow diamonds
        """

        if not self.milestones:
            return

        x = self.milestones['x_value']
        y = self.milestones['y_label']

        plt.scatter(x, y, s=120, marker="D",
                    color=self.colors, edgecolor="black", zorder=2)

    def addLegend(self):
        """Add a legend to the plot iff there are legend entries in
        the package definitions
        """

        cnt = 0
        for pkg in self.packages:
            if pkg.legend:
                cnt += 1
                idx = self.labels.index(pkg.label)
                self.barlist[idx].set_label(pkg.legend)

        if cnt > 0:
            self.legend = self.ax.legend(
                shadow=False, ncol=3, fontsize="medium")

    def render(self):
        """ Prepare data for plotting
        """

        # init figure
        plt.clf()
        self.fig, self.ax = plt.subplots(num=self.figure_num, figsize=(15, 2))
        self.ax.yaxis.grid(False)
        self.ax.xaxis.grid(False)

        self.barlist = plt.barh(self.yPos, list(self.durations),
                                left=self.start,
                                align='center',
                                height=.5,
                                alpha=1,
                                color=self.colors,
                                edgecolor=self.edge_color)
        for i, y in enumerate(self.yPos):
            plt.text((self.start[i] + self.end[i]) / 2, y,
                     str(self.packages[i].part_num), fontdict=fontdict,
                     ha='center', va='center')
        # format plot
        self.format()
        # self.addMilestones()
        plt.xlim(0, max(self.end))
        # self.addLegend()

    def show(self):

        """ Show the plot
        """

        plt.pause(0.001)
        # plt.show()

    def save(self, pg_resume):
        """ Save the plot to a file. It defaults to `img/GANTT.png`.
        :arg str saveFile: file to save to
        """
        saveFile = 'data/GanttImg/GANTT' + pg_resume + '_' + str(self.Algorithm) + '_' + 'ex' + '_' + str(
            self.ex) + '_' + '.pdf'
        plt.savefig(saveFile, bbox_inches='tight')


if __name__ == '__main__':
    g = Gantt(sample_dict)
    g.render()
    g.show()
    # g.save('img/GANTT.png')
