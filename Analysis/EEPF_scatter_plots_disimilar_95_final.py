import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.signal as sp
from scipy.stats import chisquare



#data = pd.read_csv('LM_scatter_nobayes_500.csv')
data = pd.read_csv('Total_LM_scatter_nobayes_500.csv')
data_mc = pd.read_csv('Total_LM_scatter_bayesMC_500.csv')
data_vi = pd.read_csv('Total_LM_scatter_VIbayes_500.csv')
'''
r = np.array([4,8,12,16,20,24])
rl = np.array([5,9,13,17,21,25])
ru = np.array([6,10,14,18,22,26])
rr = np.array([7,11,15, 19,23,27])
# Non-Bayesian Data Te
index = data.iloc[:,28].astype(float)

# Predicted Mean Values 
f1 = data.iloc[:,r[0]].astype(float)
f2 = data.iloc[:,r[1]].astype(float)
f3 = data.iloc[:,r[2]].astype(float)
f4 = data.iloc[:,r[3]].astype(float)
f5 = data.iloc[:,r[4]].astype(float)
f6 = data.iloc[:,r[5]].astype(float)

# Monte Carlo Dropout 
fc1 = data_mc.iloc[:,r[0]].astype(float)
fc2 = data_mc.iloc[:,r[1]].astype(float)
fc3 = data_mc.iloc[:,r[2]].astype(float)
fc4 = data_mc.iloc[:,r[3]].astype(float)
fc5 = data_mc.iloc[:,r[4]].astype(float)
fc6 = data_mc.iloc[:,r[5]].astype(float)

# Variational Inference
fv1 = data_vi.iloc[:,r[0]].astype(float)
fv2 = data_vi.iloc[:,r[1]].astype(float)
fv3 = data_vi.iloc[:,r[2]].astype(float)
fv4 = data_vi.iloc[:,r[3]].astype(float)
fv5 = data_vi.iloc[:,r[4]].astype(float)
fv6 = data_vi.iloc[:,r[5]].astype(float)
'''
'''
# Good Data similar!
# non_bayesian
nbm = np.array([0.20613305, 0.50276165, 0.50119062, 0.6604103,  1.65610633, 5.61284775,3.20165479])
nbl = np.array([0.15054377, 0.45485725, 0.3301524,  0.64300798, 1.54673555, 5.31718698, 3.12825295])
nbh = np.array([0.25990531, 0.55249339, 0.66004653, 0.67887192, 1.77270402, 5.93418609, 3.27272377])
# bayesian MC
bmcm = np.array([0.17674618, 0.49071634, 0.4505694,  0.66812886, 1.68448049, 5.74836025, 3.21063321])
bmcl = np.array([0.12083995, 0.44978812, 0.32003106, 0.65003207, 1.58922045, 5.38021566, 3.09015034])
bmch = np.array([0.23369988, 0.53193677, 0.57840714, 0.68543723, 1.76852655, 6.11697461, 3.34159832])
# bayesian Vi
bvm = np.array([0.18067206, 0.49095522, 0.4772741,  0.65382731, 1.63611815, 5.55768526,3.14004957])
bvl = np.array([0.11855429, 0.44143269, 0.29583846, 0.63291286, 1.52103693, 5.20614821, 3.08891363])
bvh = np.array([0.24058822, 0.54749943, 0.66303919, 0.67546961, 1.75915471, 5.8909922, 3.18951483])
# true 
tt = np.array([0.15057386, 0.46788141, 0.37199432, 0.67340486, 1.69435069, 5.72064905, 3.19043953])
'''
'''
# dissimilar one (not this one)
# non_bayesian
nbm = np.array([0.40834134, 0.63817183, 1.13250275, 0.68163975, 1.69153075, 4.95851749, 2.87741637])
nbl = np.array([0.35914404, 0.58971654, 0.96089583, 0.66181996, 1.59049483, 4.67356089, 2.81283795])
nbh = np.array([0.45740433, 0.68414745, 1.31383421, 0.70182898, 1.79573653, 5.23654893, 2.94682229])
# bayesian MC
bmcm = np.array([0.28247624, 0.55705837, 0.87233722, 0.71712179, 1.89382294, 5.91593662, 3.12890257])
bmcl = np.array([0.20938805, 0.50406089, 0.66951021, 0.69652166, 1.73266124, 5.37286375, 2.95483085])
bmch = np.array([0.34973308, 0.60934013, 1.09056023, 0.74092296, 2.03700172, 6.38354698, 3.29554202])
# bayesian Vi
bvm = np.array([0.39905276, 0.64061619, 1.15244517, 0.7061812,  1.87517464, 5.72410762, 2.94181621])
bvl = np.array([0.37781827, 0.62565229, 1.07110057, 0.66998262, 1.61057045, 5.01303375, 2.79707843])
bvh = np.array([0.42140106, 0.65668859, 1.24001843, 0.74149912, 2.15029567, 6.43660062, 3.07777586])
# true 
#tt = np.array([)
'''


# dissimilar final
# non_bayesian
nbm = np.array([0.2978504,  0.5895058,  0.9508944,  0.64369492, 1.890888, 6.08687046, 3.14571937])
nblo = np.array([0.24915901, 0.5469997,  0.77163387, 0.62591783, 1.77825036, 5.79535114, 3.07767839])
nbho = np.array([0.35207807, 0.63373568, 1.1281217,  0.66022686, 1.99079446, 6.3966354, 3.21816306])
nbl = nbm - ((nbho - nbm)/2)
nbh = nbm + ((nbho - nbm)/2)

# bayesian MC
bmcm = np.array([0.2960552,  0.56797192, 0.91116277, 0.69511618, 1.83497538, 5.87874525, 3.17657526])
bmclo = np.array([0.18332417, 0.47567198, 0.60767441, 0.65477075, 1.63629825, 5.20435807, 2.98831272])
bmcho = np.array([0.38811567, 0.64378626, 1.23113154, 0.73237779, 2.03095557, 6.47448422, 3.37609256])

bmcl = bmcm - ((bmcho - bmcm)/2)
bmch = bmcm + ((bmcho - bmcm)/2)


# bayesian Vi
bvm = np.array([0.33547666, 0.59606597, 1.09475095, 0.71407412, 1.99346334, 6.2151235, 3.08538195])
bvlo = np.array([0.26621647, 0.54554347, 0.83846774, 0.64123814, 1.39969651, 4.86625228, 2.80213431])
bvho = np.array([0.40112246, 0.64671743, 1.35114493, 0.78682065, 2.52081983, 7.44820292, 3.34462983])


bvl = bvm - ((bvho - bvm)/2)
bvh = bvm + ((bvho - bvm)/2)






def funcc(x, a,b,c):
    it1 = (c**-1.5)*a*np.exp(-(x/c)**b)
    return (it1*np.sqrt(x))

def pred(x,p1,p2,p3,p4,p5,p6):
    result = list()
    for i in x:
        if i <= 5:
            result.append(funcc(i,p1,p2,p3))
        else:
            result.append(funcc(i,p4,p5,p6))
    return result

def sav(x):
    return sp.savgol_filter(x, 1051,2)

def savt(x):
    return sp.savgol_filter(x, 451,2)

n = 1
E = np.linspace(1,22,10000)

'''
fig = plt.gcf()
plt.grid(True)
#fig.set_size_inches(10, 10)

plt.rcParams.update({'font.size': 13})

plt.plot(E, sav(pred(E,nbm[0],nbm[1],nbm[2],nbm[3],nbm[4],nbm[5],)), 'r', linewidth=3)


#plt.plot(E, sav(pred(E,nbl[0],nbm[1],nbm[2], nbl[3],nbm[4],nbl[5])), 'b', linewidth=2)
#plt.plot(E, sav(pred(E,nbm[0],nbl[1],nbl[2], nbm[3],nbl[4],nbm[5])), 'b', linewidth=2)

plt.fill_between(E,sav(pred(E,nbl[0],nbm[1],nbm[2], nbl[3],nbm[4],nbl[5])),
                 sav(pred(E,nbm[0],nbl[1],nbl[2], nbm[3],nbl[4],nbm[5])), alpha=1)

#plt.plot(E, sav(pred(E,nblo[0],nbm[1],nbm[2], nblo[3],nbm[4],nblo[5])), 'g', linewidth=2)
#plt.plot(E, sav(pred(E,nbm[0],nblo[1],nblo[2], nbm[3],nblo[4],nbm[5])), 'g', linewidth=2)

plt.fill_between(E,sav(pred(E,nblo[0],nbm[1],nbm[2], nblo[3],nbm[4],nblo[5])),
                 sav(pred(E,nbm[0],nblo[1],nblo[2], nbm[3],nblo[4],nbm[5])), alpha=0.2, color= 'c')


plt.legend(['Predicted $f_p$', '68% Confidence Interval', '95% Confidence Interval'])



#plt.plot(E, sav(pred(E,nbl[0],nbl[1],nbl[2],nbl[3],nbl[4],nbl[5],)), 'b--', linewidth=2)
#plt.plot(E, sav(pred(E,nbh[0],nbh[1],nbh[2],nbh[3],nbh[4],nbh[5],)),'b--', linewidth=2)
#plt.plot(E, savt(pred(E,tt[0],tt[1],tt[2],tt[3],tt[4],tt[5],)), 'g')
plt.yscale('log')
plt.text(0.5, 0.25, "(a) Non-Bayes", fontsize=13)
#plt.xlabel('Energy (eV)', fontsize=13)
plt.ylabel('$f_p$ $(eV^{-3/2})$', fontsize=13)
plt.grid(True)
plt.ylim(1e-3,2*2e-1)
plt.xlim(0,21)
#ax.axes.xaxis.set_ticks([])
plt.grid(True)
plt.tick_params(direction='in')
plt.tick_params(which='minor', direction='in')
#plt.legend(['Predicted $f_p(E)$', '97.50% prec.', '2.50% prec.', 'Observed $f_p(E)$ '])
#plt.legend(['Predicted $f_p$', '97.5% perc.', '2.5% perc.','Measured $f_p$' ])
#plt.legend(['Predicted $f_p$', '97.5% perc.', '2.5% perc.'])
'''

'''
fig = plt.gcf()
plt.grid(True)
plt.rcParams.update({'font.size': 13})
#fig.set_size_inches(10, 10)

plt.plot(E, sav(pred(E,bvm[0],bvm[1],bvm[2],bvm[3],bvm[4],bvm[5],)), 'r', linewidth=3)

#plt.plot(E, sav(pred(E,bvl[0],bvm[1],bvm[2],bvl[3],bvm[4],bvl[5])), 'b', linewidth=2)
#plt.plot(E, sav(pred(E,bvm[0],bvl[1],bvl[2],bvm[3],bvl[4],bvm[5])), 'b', linewidth=2)

plt.fill_between(E,sav(pred(E,bvl[0],bvm[1],bvm[2],bvl[3],bvm[4],bvl[5])),
                 sav(pred(E,bvm[0],bvl[1],bvl[2],bvm[3],bvl[4],bvm[5])), alpha=1
                 )

#plt.plot(E, sav(pred(E,bvlo[0],bvm[1],bvm[2],bvlo[3],bvm[4],bvlo[5])), 'g', linewidth=2)
#plt.plot(E, sav(pred(E,bvm[0],bvlo[1],bvlo[2],bvm[3],bvlo[4],bvm[5])), 'g', linewidth=2)

plt.fill_between(E,sav(pred(E,bvlo[0],bvm[1],bvm[2],bvlo[3],bvm[4],bvlo[5])),
                 sav(pred(E,bvm[0],bvlo[1],bvlo[2],bvm[3],bvlo[4],bvm[5])), alpha=0.2, color='c'
                 )


plt.legend(['Predicted $f_p$', '68% Confidence Interval', '95% Confidence Interval'])




#plt.plot(E, sav(pred(E,bvl[0],bvl[1],bvl[2],bvl[3],bvl[4],bvl[5],)), 'b--', linewidth=2)
#plt.plot(E, sav(pred(E,bvh[0],bvh[1],bvh[2],bvh[3],bvh[4],bvh[5],)),'b--', linewidth=2)
#plt.plot(E, savt(pred(E,tt[0],tt[1],tt[2],tt[3],tt[4],tt[5],)), 'g')
plt.yscale('log')
plt.text(0.5, 0.25, "(b) Bayes (VI)", fontsize=13)
#plt.xlabel('Energy (eV)', fontsize=13)
plt.ylabel('$f_p$ $(eV^{-3/2})$', fontsize=13)
plt.grid(True)
plt.ylim(1e-3,2*2e-1)
plt.xlim(0,21)
plt.tick_params(direction='in')
plt.tick_params(which='minor', direction='in')
#plt.legend(['Predicted $f_p$', '97.5% perc.', '2.5% perc.'], fontsize=13)
#plt.legend(['Predicted $f_p$', '97.5% perc.', '2.5% perc.','Measured $f_p$' ])

#plt.legend(['Predicted $f_p$', '97.50% perc.', '2.50% perc.'])
'''



fig = plt.gcf()
plt.grid(True)
plt.rcParams.update({'font.size': 13})
#fig.set_size_inches(10, 10)


#plt.plot(E, sav(pred(E,bmcl[0],bmcm[1],bmcm[2],bmcl[3],bmcm[4],bmcl[5])), 'b', linewidth=3)
#plt.plot(E, sav(pred(E,bmcm[0],bmcl[1],bmcl[2],bmcm[3],bmcl[4],bmcm[5])), 'b', linewidth=3)

plt.plot(E, sav(pred(E,bmcm[0],bmcm[1],bmcm[2],bmcm[3],bmcm[4],bmcm[5],)), 'r', linewidth=3)

plt.fill_between(E,sav(pred(E,bmcl[0],bmcm[1],bmcm[2],bmcl[3],bmcm[4],bmcl[5])),
                       sav(pred(E,bmcm[0],bmcl[1],bmcl[2],bmcm[3],bmcl[4],bmcm[5])),
                       alpha=1)

#plt.plot(E, sav(pred(E,bmclo[0],bmcm[1],bmcm[2],bmclo[3],bmcm[4],bmclo[5])), 'g', linewidth=3)
#plt.plot(E, sav(pred(E,bmcm[0],bmclo[1],bmclo[2],bmcm[3],bmclo[4],bmcm[5])), 'g', linewidth=3)

plt.fill_between(E,sav(pred(E,bmclo[0],bmcm[1],bmcm[2],bmclo[3],bmcm[4],bmclo[5])),
                 sav(pred(E,bmcm[0],bmclo[1],bmclo[2],bmcm[3],bmclo[4],bmcm[5])), alpha=0.2, color='c')

plt.legend(['Predicted $f_p$', '68% Confidence Interval', '95% Confidence Interval'])
#plt.plot(E, sav(pred(E,bmcl[0],bmcl[1],bmcl[2],bmcl[3],bmcl[4],bmcl[5],)), 'b--', linewidth=2)
#plt.plot(E, sav(pred(E,bmch[0],bmch[1],bmch[2],bmch[3],bmch[4],bmch[5],)),'b--', linewidth=2)
#plt.plot(E, savt(pred(E,tt[0],tt[1],tt[2],tt[3],tt[4],tt[5],)), 'g')
plt.yscale('log')
plt.text(0.5, 0.25, "(c) Bayes (MCD)", fontsize=13)
plt.xlabel('Energy (eV)', fontsize=13)
plt.ylabel('$f_p $ $(eV^{-3/2})$', fontsize=13)
plt.grid(True)
plt.ylim(1e-3,2*2e-1)
plt.xlim(0,21)
plt.tick_params(direction='in')
plt.tick_params(which='minor', direction='in')
#plt.legend(['Predicted Mean $f_p(E)$', '97.50% prec.', '2.50% prec.', 'Observed $f_p(E)$ '])
#plt.legend(['Predicted $f_p$', '97.5% perc.', '2.5% perc.'])
#plt.legend(['Predicted $f_p$', '97.5% perc.', '2.5% perc.','Measured $f_p$' ])


fig.savefig('EEPFS_dis_bayesmcd_fico.png', format='png', dpi=600)


#fig.savefig('EEPFS_bayesmc_off2.png', format='png', dpi=600)
