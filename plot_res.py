import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_res(data, themes, seasons):
    N = len(themes)
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
    plt.savefig(f'results/test/total.png', dpi=300)

parts = ['test', 'val']
for part in parts:
    themes = ['streak', 'standing', 'average', 'double']
    seasons = ["all", "bens", "carlos", "joels", "dans", "oscars"]
    data = {season: [] for season in seasons}
    for theme in themes:
        for season in seasons:
            try:
                res = json.load(open(f'./results/{part}/{theme}-{season}.json', 'r'))
                data[season].append(res['mean_mf1']*100)
            except:
                data[season].append(0)

    # plot_res(data, themes, seasons)
    df = pd.DataFrame(data, index=themes, columns=seasons)
    plt.rcParams['font.size'] = 22
    df.plot(kind='bar', figsize=(15, 10))
    plt.tight_layout()
    plt.savefig(f'results/{part}/total.png', dpi=300)
    # plt.show()
