import json
import numpy as np
import matplotlib.pyplot as plt

def plot_res(res_dict, theme, metric):

    N = 6
    ind = np.arange(N)
    width = 0.10

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    ftr_keys = list(res_dict['rf'].keys())

    yvals = [res_dict[clf][ftr_keys[0]]*100 for clf in clfs]
    rects1 = ax.bar(ind, yvals, width, color='r')
    zvals = [res_dict[clf][ftr_keys[1]]*100 for clf in clfs]
    rects2 = ax.bar(ind+width, zvals, width, color='g')
    kvals = [res_dict[clf][ftr_keys[2]]*100 for clf in clfs]
    rects3 = ax.bar(ind+width*2, kvals, width, color='b')
    mvals = [res_dict[clf][ftr_keys[3]]*100 for clf in clfs]
    rects4 = ax.bar(ind+width*3, mvals, width, color='y')

    ax.set_ylabel(f'{metric} (%)')
    ax.set_xlabel('Classifiers')
    ax.title.set_text(f'{theme} {metric}')
    ax.set_xticks(ind+width*1.5)
    ax.set_xticklabels([x.upper() for x in clfs])
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), tuple(ftr_keys), ncol=4, loc='upper center')
    ax.set_ylim(0, 110)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    plt.tight_layout()
    plt.rcParams.update({'font.size': 20})
    plt.savefig(f'plots/{theme}/{metric}.png', dpi=300)


for theme in ['streak', 'standing']:

    clfs = ['rf', 'if', 'svm', 'bert', 'pet', 'tpot']
    ftrs = ['text', 'num']

    f1s, accs, mf1s, precs, recs, fbetas = {}, {}, {}, {}, {}, {}
    for clf in clfs:
        f1s[clf], accs[clf], mf1s[clf], precs[clf], recs[clf], fbetas[clf] = {}, {}, {}, {}, {}, {}

        for ftr in ftrs:
            for do_down in ['yes', 'no']:
                empty_js = {'mf1': 0, 'acc': 0, 'f1': 0, 'prec': 0, 'rec': 0, 'fbeta': 0}
                if clf == 'bert' and ftr == 'num':
                    clf_results = empty_js
                elif clf == 'if' and do_down == 'yes':
                    clf_results = empty_js
                elif (clf == 'pet' and ftr == 'num') or (clf == 'pet' and do_down == 'yes'):
                    clf_results = empty_js
                elif (clf == 'tpot' and ftr == 'text') or (clf == 'tpot' and do_down == 'no'):
                    clf_results = empty_js
                else:
                    js = json.load(open(f"results/{theme}/{clf}-{ftr}-down_{do_down}.json"))
                    clf_results = js['clf_results']

                ft = 'Text' if ftr == 'text' else 'Num'
                dyn = 'DY' if do_down == 'yes' else 'DN'
                f1s[clf][f'{ft}-{dyn}'] = clf_results['f1']
                accs[clf][f'{ft}-{dyn}'] = clf_results['acc']
                mf1s[clf][f'{ft}-{dyn}'] = clf_results['mf1']
                precs[clf][f'{ft}-{dyn}'] = clf_results['prec']
                recs[clf][f'{ft}-{dyn}'] = clf_results['rec']
                fbetas[clf][f'{ft}-{dyn}'] = clf_results['fbeta']

    # mf1s = {clf: {ftr: round(mf1s[clf][ftr]/100, 2) for ftr in mf1s[clf]} for clf in mf1s}
    print(f"{theme} F1: \n{f1s}\n\n")
    print(f"{theme} Acc: \n{accs}\n\n")
    print(f"{theme} MF1: \n{mf1s}\n\n")
    print(f"{theme} Prec: \n{precs}\n\n")
    print(f"{theme} Rec: \n{recs}\n\n")
    print(f"{theme} Fbeta: \n{fbetas}\n\n")

    plot_res(accs, theme.capitalize(), 'Accuracy')
    plot_res(f1s, theme.capitalize(), 'F1')
    plot_res(precs, theme.capitalize(), 'Precision')
    plot_res(recs, theme.capitalize(), 'Recall')
    plot_res(fbetas, theme.capitalize(), 'Fbeta')
    # plot_res(mf1s, theme.capitalize(), 'Macro F1')
