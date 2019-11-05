import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 8,7
import math

def plot_pdf(ieta=0, ipt=0):
    print("plot_pdf()")
    pdf = np.load('pdf.npz')
    res, res_err = pdf['arr_0'], pdf['arr_1']
    eta_bins, pt_bins = pdf['arr_2'], pdf['arr_3']
    y_vals, bT_vals = pdf['arr_4'], pdf['arr_5']
    plt.figure()
    fig, ax = plt.subplots()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'p' ]
    fmts = ['o', 'v', '^', '>', '<', '.', '-']
    plot_iy = range(min(y_vals.size, 5))
    plot_iy = [0,1,2,3,4]
    for iy in plot_iy:
        ax.errorbar(bT_vals, res[ieta,ipt,iy,:], xerr=0, yerr=res_err[ieta,ipt,iy,:], #fmt=fmts[iy], 
                    color=colors[iy], label='pdf vs bT, y='+'{:04.3f}'.format(y_vals[iy]) )        
    legend = ax.legend(loc='best', shadow=False, fontsize='x-large')
    plt.axis([-0.01, 1.0 , 0.0,  np.max(res[ieta,ipt,0,:])*2.0 ])
    plt.grid(True)
    plt.xlabel('$b_{T}$ value', fontsize=20)
    plt.ylabel('pdf (xexp factorized)', fontsize=20)
    plt.title('pdf vs bT, $p_{T}\in[$'+'{:04.1f}'.format(pt_bins[ipt])+','+'{:04.1f}'.format(pt_bins[ipt+1])+']', fontsize=20)
    plt.savefig('pdf_vs_bT_ieta'+'{}'.format(ieta)+'_ipt'+'{}'.format(ipt)+'.png')
    plt.close()

if __name__ == '__main__':
    for i in range(6):
        plot_pdf(ipt=i)
