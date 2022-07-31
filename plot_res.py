import json
import numpy as np
import matplotlib.pyplot as plt

def plot_res(data, themes):
    N = 4
    ind = np.arange(N)
    width = 0.2
    fig = plt.figure(figsize=(15, 10))
    plt.rcParams['font.size'] = 22
    ax = fig.add_subplot(111)
    yvals = data['all']
    rects1 = ax.bar(ind, yvals, width, color='b')
    zvals = data['bens']
    rects2 = ax.bar(ind+width, zvals, width, color='r')

    ax.set_ylabel('Macro F1')
    ax.set_xticks(ind+width)
    ax.set_xticklabels([i.capitalize() for i in themes])
    ax.legend((rects1[0], rects2[0]), list(data.keys()))

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.ylim(0, 100)
    plt.tight_layout()
    plt.rcParams['font.size'] = 22
    plt.savefig(f'results/total.png', dpi=300)

themes = ['streak', 'standing', 'average', 'double']
data = {"all": [], "bens": []}
for theme in themes:
    res_all = json.load(open(f'./results/{theme}-all.json', 'r'))
    res_bens = json.load(open(f'./results/{theme}-bens.json', 'r'))
    data['all'].append(res_all['clf_results']['mf1']*100)
    data['bens'].append(res_bens['clf_results']['mf1']*100)

plot_res(data, themes)
