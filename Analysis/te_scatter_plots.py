import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

reg = LinearRegression()


#data = pd.read_csv('LM_scatter_nobayes_500.csv')
data = pd.read_csv('Total_LM_scatter_nobayes_500.csv')
data_mc = pd.read_csv('Total_LM_scatter_bayesMC_500.csv')
data_vi = pd.read_csv('Total_LM_scatter_VIbayes_500.csv')

r = np.array([0,1,2,3])
# Non-Bayesian Data Te
index = data.iloc[:,28].astype(float)
te_pred = data.iloc[:,r[0]].astype(float)
te_predl = data.iloc[:,r[1]].astype(float)
te_predh = data.iloc[:,r[2]].astype(float)
te_true = data.iloc[:,r[-1]].astype(float)
# Monte Carlo Dropout Te
te_pred_mc = data_mc.iloc[:,r[0]].astype(float)
te_predl_mc = data_mc.iloc[:,r[1]].astype(float)
te_predh_mc = data_mc.iloc[:,r[2]].astype(float)
te_true_mc = data_mc.iloc[:,r[-1]].astype(float)
# Variation Inference Te
te_pred_vi = data_vi.iloc[:,r[0]].astype(float)
te_predl_vi = data_vi.iloc[:,r[1]].astype(float)
te_predh_vi = data_vi.iloc[:,r[2]].astype(float)
te_true_vi = data_vi.iloc[:,r[-1]].astype(float)

#reg.fit(te_true,te_pred)

n = 60
fig = plt.gcf()
#fig.set_size_inches(6,6)
plt.grid(True)
plt.scatter(te_true,te_pred, s=n, marker='o', c='b',alpha=0.8)
plt.scatter(te_true,te_pred_mc, s=n, marker='o', c='r',alpha=0.8)
plt.scatter(te_true,te_pred_vi, s=n, marker='o', c='g',alpha=0.8)

# Mean Absolute Percentage Error
def error(pred,true):
    return np.average((abs(true-pred)/true)*100)

no_bayes = r2_score(te_true, te_pred)
bayes_mc = r2_score(te_true, te_pred_mc)
bayes_vi = r2_score(te_true, te_pred_vi)

er1 = error(te_pred,te_true)
er_mc = error(te_pred_mc,te_true_mc)
er_vi = error(te_pred_vi,te_true_vi)

print('R2 score: no_bayes', no_bayes)
print('R2 score: bayes MC', bayes_mc)
print('R2 score: bayes VI', bayes_vi)
print('Average percentage error: no_bayes', er1)
print('Average percentage error: MC', er_mc)
print('Average percentage error: bayes VI', er_vi)
plt.rcParams.update({'font.size' : 13})  
plt.legend(['R2: {0:.2f} and MAPE: {1:.2f}% - Non-Bayes'.format(no_bayes,er1),
            'R2: {0:.2f} and MAPE: {1:.2f}% - Bayes (MCD)'.format(bayes_mc, er_mc),
            'R2: {0:.2f} and MAPE: {1:.2f}% - Bayes (VI)'.format(bayes_vi,er_vi)], fontsize=13)
#plt.xlabel('Observed ${T_e (eV)}$')
#plt.ylabel('Predicted Mean ${T_e (eV)}$')
plt.xlabel('Measured ${T_e}$ (eV) ', fontsize=13)
plt.ylabel('Predicted ${T_e}$ (eV)', fontsize=13)
plt.xlim(2,4.5)
plt.ylim(2,4.5)




fig.savefig('Te_final.png', format='png', dpi=600)





