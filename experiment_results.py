

import matplotlib
import numpy as np

from os import listdir
from os.path import isfile, join
import os
import re

import experiment
import matplotlib.pyplot as plt

class ExperimentResults:
    def getFlagValue(self, name):
        if name in self.flags.keys():
            return self.flags[name]
        return experiment.Experiment.FLAGS_DEFAULTS[name]

    def __init__(self, label, flags):
        self.flags = flags
        self.trainError = []
        self.testError = []
        self.epochTimes = []
        self.epochsDone = 0
        self.errors_b4_merge = []
        self.errors_after_merge = []
        self.cg_f_vals = []
        self.cg_grad_norms = []
        self.sesop_indeces = []
        self.h_of_alphas = {}
        self.alphas = {}
        self.label = label

    def buildLabel(self, flag_names_to_use_in_label):
        res = ''
        for flag_name in flag_names_to_use_in_label:
            res += '/' + flag_name + '=' + str(self.getFlagValue(flag_name))
        return res

    def getBestTrainError(self):
        return min([l for l in self.trainError if l > 0])

    def getBestTestError(self):
        return min([l for l in self.testError if l > 0])

    def plotTrainError(self, l=(0, 100), flag_names_to_use_in_label=None):
        plt.title('Train Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        full_label = 'train(' + self.label + ')'
        if flag_names_to_use_in_label is not None:
            full_label = self.buildLabel(flag_names_to_use_in_label)
        plt.plot(range(len(self.trainError[l[0]:l[1]])), self.trainError[l[0]:l[1]], '-', label=full_label)
        plt.legend()

    def plotTestError(self, l=(0, 100), flag_names_to_use_in_label=None):
        plt.title('Test Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')

        full_label = 'test(' + self.label + ')'
        if flag_names_to_use_in_label is not None:
            full_label = self.buildLabel(flag_names_to_use_in_label)

        #print 'Test min error is ' + str(1 - min(self.testError[l[0]:l[1]]))

        plt.plot(range(len(self.testError[l[0]:l[1]])), self.testError[l[0]:l[1]], '-', label=full_label)
        # plt.plot(range(self.testError[:l].size), self.testError[:l], 'o')
        plt.legend()


    def plotTrainErrorAroundMerge(self, l=(0,100)):

        y = []
        for i in range(1, len(self.errors_b4_merge)):
            if i < l[0]:
                continue
            if i > l[1]:
                break

            y.append(self.errors_b4_merge[i])
            plt.axvline(x=len(y)-1, ls='-', color='g') #mark begining
            try:
                y.append(self.errors_after_merge[i])
            except:
                y.append(10)

            plt.axvline(x=len(y)-1, ls='--', color='r')  #mark end

        x = range(len(y))
        print len(x)
        print len(y)
        plt.plot(x, y)

    def getSesopIdxes(self, l):
        res = []
        s = 0
        for idx in self.sesop_indeces[1:]:
            s += idx
            if s > l[1]: break
            res.append(s-1)

    def plotH(self, l = (0, 100), iter=0, history_vec_idx=0, label=''):
        plt.title('h - The function to perform line search on. iter = ' + str(iter))
        plt.xlabel('alpha')
        plt.ylabel('h')

        h = self.h_of_alphas[history_vec_idx][iter]
        a = self.alphas[history_vec_idx][iter]
        l = (l[0], min(l[1], h.size))
        plt.plot(a[l[0]:l[1]], h[l[0]:l[1]], '-', label=label)
        #plt.plot(range(h[l[0]:l[1]].size), h[l[0]:l[1]], 'o')

    def plotCGFValsByIdx(self, i):
        plt.title('CG f vals of the ' + str(i) + ' SESOP run')
        plt.xlabel('Iteration')
        plt.ylabel('value')

        l = self.getSesopIterationIdxes(i)

        l = (l[0], min(l[1], self.cg_f_vals.size))
        plt.plot(range(l[0], l[1] + 1), self.cg_f_vals[l[0]:l[1] + 1], '-', label=self.label)
        self.addSesopVLines(l)

    def plotCGFVals(self, l = (0, 100)):
        plt.title('CG f vals')
        plt.xlabel('Iteration')
        plt.ylabel('value')

        l = (l[0], min(l[1], self.cg_f_vals.size))
        plt.plot(range(l[0], l[1]), self.cg_f_vals[l[0]:l[1]], '-', label=self.label)
        self.addSesopVLines(l)
        #plt.plot(range(self.testError[:l].size), self.testError[:l], 'o')

    #get CG iteration index and return sesop iteration idx
    def getSesopIterationIdxFromCgIterationIdx(self, cg_i):
        for i in range(len(self.sesop_indeces)):
            res = self.getSesopIterationIdxes(i)
            if res[0] <= cg_i and res[1] >= cg_i:
                return i
        return None

    #return the CG indexes for the ith sesop iteration
    def getSesopIterationIdxes(self, i):
        start = 0
        end = 0
        count = 0
        for idx in self.sesop_indeces[1:]:
            start = end + 1
            end += idx
            count = count + 1
            if count >= i:
                break
        return [int(start), int(end)]



    def addSesopVLines(self, l):
        #s = 0
        for i in range(len(self.sesop_indeces)):
            idxes = self.getSesopIterationIdxes(i)
            if idxes[1] < l[0]: continue
            if idxes[0] > l[1]: break
            #print idxes
            plt.axvline(x=idxes[0], ls='-', color='r')
            plt.axvline(x=idxes[1], ls='--', color='r')

        #for idx in self.sesop_indeces[1:]:
        #    s += idx
        #    if s < l[0]: continue
        #    if s > l[1]: break
        #    plt.axvline(x=s-1, color='r')


    def plotCGGradNormsByIdx(self, i):
        plt.title('CG grad norm of the ' + str(i) + ' SESOP run')
        plt.xlabel('Iteration')
        plt.ylabel('value')

        l = self.getSesopIterationIdxes(i)

        l = (l[0], min(l[1], self.cg_grad_norms.size))

        #print 'l = ' + str(l)
        #print 'range(l[0], l[1]) = ' + str(range(l[0], l[1]))
        #print 'self.cg_grad_norms[l[0]:l[1]] = ' + str(self.cg_grad_norms[l[0]:l[1]])

        plt.plot(range(l[0], l[1] + 1), self.cg_grad_norms[l[0]:l[1] + 1], '-', label=self.label)
        self.addSesopVLines(l)

    def plotCGGradNorms(self, l = (0, 100)):
        plt.title('CG grad norm')
        plt.xlabel('Iteration')
        plt.ylabel('Norm')

        l = (l[0], min(l[1], self.cg_grad_norms.size))
        plt.plot(range(l[0], l[1]), self.cg_grad_norms[l[0]:l[1]], '-', label=self.label)

        self.addSesopVLines(l)

        #plt.plot(range(self.testError[:l].size), self.testError[:l], 'o')
        plt.legend()


    def plotTimeToAcc(self, l = 100):
        plt.title('Time to Error')
        plt.xlabel('Error')
        plt.ylabel('Epochs')
        xs = sorted(self.testError[1:l])
        ys = []
        for x in xs:
            #find the first x' that is smaller than x
            for i in range(len(xs)):
                if self.testError[1:l][i] <= x:
                    ys.append(i)
                    break

        plt.yscale('log')
        plt.plot(xs, ys, '-', label=self.label)
        #plt.plot(range(self.testError[:l].size), self.testError[:l], 'o')
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))




#given two exprs, return a list that contain the different flags
def experiment_diff(es):
    res = []
    for flag_name in experiment.Experiment.FLAGS_DEFAULTS.keys():
        for e in es:
            if es[1].getFlagValue(flag_name) != e.getFlagValue(flag_name):
                res.append(flag_name)
                break
    return res

class ColorCycle:
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    def __init__(self):
        self.i = 0

    def get(self):
        res = ColorCycle.colors[self.i]
        self.i = (self.i + 1)%len(ColorCycle.colors)
        return res


class ExperimentComperator:
    #experiments is a dictionary of experiments
    def __init__(self, experiments):
        self.experiments = experiments
        self.x_logscale = False
        self.y_logscale = False

    def set_x_logscale(self, val=True):
        self.x_logscale = val
        return self
    def set_y_logscale(self, val=True):
        self.y_logscale = val
        return self

    def apply_figure_attributes(self):
        if self.x_logscale:
            plt.xscale('log')

        if self.y_logscale:
            plt.yscale('log')

    def getBestTrainError(self, filter=None):
        experiments = {k:v for k,v in self.experiments.items() if filter is None or filter(v)}
        best_e = experiments.values()[0]
        for i in experiments.keys():
            if experiments[i].results.getBestTrainError() < best_e.results.getBestTrainError():
                best_e = experiments[i]
        return best_e

    #group_by is a name of a flag, its meaning is that we will present one plot
    #for each flag value present in e.
    def compare(self, group_by='h', error_type='train', filter=None):
        import math
        from matplotlib.font_manager import FontProperties


        e = {k:v for k,v in self.experiments.items() if filter is None or filter(v)}
        r = {}
        for k in e.keys():
            r[k] = e[k].results

        #group ny flag
        apeared_values = {}
        for k in r.keys():
            if e[k].getFlagValue(group_by) not in apeared_values.keys():
                apeared_values[e[k].getFlagValue(group_by)] = [] #this value has apeared
            apeared_values[e[k].getFlagValue(group_by)].append(e[k])



        #Plot train error for each group:
        for val in sorted(apeared_values.keys()):
            expreiments = apeared_values[val]
            plt.figure(group_by + str(expreiments[0].getFlagValue(group_by)), figsize=(10,8))
            self.apply_figure_attributes()

            print 'val = ' + str(val) + ', expreiments = ' + str(expreiments)
            for expr in expreiments:
                diff = experiment_diff(expreiments)

                #On the label, show the group_by value, and the diff values
                if error_type == 'test':
                    expr.results.plotTestError((0,100), diff + [group_by])
                else:
                    expr.results.plotTrainError((0,100), diff + [group_by])
            #plt.legend(loc='center left', bbox_to_anchor=(0.4, 1))

            fontP = FontProperties()
            fontP.set_size('small')
            #plt.legend("title", prop=fontP)
            plt.legend(prop=fontP)

        #Now plot bars for the best score achived for each experiemnt value
        fig = plt.figure(figsize=(10,8))
        #fig = plt.figure()
        ax = plt.subplot(111)

        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

        pos = 0
        width = 0.35       # the width of the bars
        nextLegendHeight = 1
        #for each value that apeared, we take all the experiments
        cycler = ColorCycle()
        for val in apeared_values.keys():
            expreiments = apeared_values[val]
            diff = experiment_diff(expreiments)
            #now we want to add a bar for each of these
            rects = []
            labels = []
            for expr in expreiments:
                final_label = expr.results.buildLabel(diff + [group_by])
                if error_type == 'test':
                    rect = ax.bar(pos, expr.results.getBestTestError(), width,\
                              color=cycler.get())
                else:
                    rect = ax.bar(pos, expr.results.getBestTrainError(), width,\
                              color=cycler.get())

                rects.append(rect)
                labels.append(final_label)
                pos += width

            fontP = FontProperties()
            fontP.set_size('small')
            #plt.legend("title", prop=fontP)
            fig.gca().add_artist(ax.legend(rects, labels, loc=1, bbox_to_anchor=(1, nextLegendHeight), prop=fontP))
            nextLegendHeight -= 0.1
            pos += 1
        #plt.legend()
        #plt.xscale('log')
        plt.title('Best error after 100 epochs')
        plt.xlabel(group_by)
        plt.ylabel('Best error in 100 epochs')
        plt.grid(True)


