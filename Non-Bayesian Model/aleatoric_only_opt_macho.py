import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
import os, glob
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Dropout, Dense
from tensorflow.keras.models import Sequential, Model, load_model
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf
import scipy.signal as sp
from tqdm import tqdm_notebook as tqdm

'''
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "stars_combined.csv", index=False, encoding='utf-8-sig')
'''
#from keras import backened as K
# Import the Data from the appropriate path directory. 
# Importing ML input/Output file
data = pd.read_csv('Processed_Data.csv')
# Inputs 
p1 = data['# p_696_57']
p2 = data[' p_706_73']
p3 = data['p_738_45']
p4 = data['p_750_42']
p5 = data[' p_751_52']
p6 = data[' p_763_55']
p7 = data[' p_772_44']
p8 = data[' p_794_90']
p9 = data[' p_800_70']
p10 = data[' p_801_57']
p11 = data[' p_810_43']
p12 = data['p_811_58']
p13 = data[' p_826_49']
p14 = data[' p_840_90']
p15 = data[' p_842_53']
p16 = data[' p_852_21']
p17 = data[' p_912_33']
p18 = data[' p_922_48']
p19 = data['p_965_80']
pex = data[' He_Flow_Rate']
pexx = data['H2_Flow_Rate (sccm)']
# Outputs
fit_10 = data['fit10']
fit_11 = data[' fit11']
fit_12 = data[' fit12']
fit_13 = data[' fit13']
fit_14 = data[' fit14']
fit_15 = data[' fit15']
te = data['Electron Temperature (eV)']



def norm(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

fitx_10 = norm(fit_10)
fitx_11 = norm(fit_11)
fitx_12 = norm(fit_12)
fitx_13 = norm(fit_13)
fitx_14 = norm(fit_14)
fitx_15 = norm(fit_15)
tex = norm(te)

def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


def my_dist(params): 
  return tfd.Normal(loc=params[...,:7], scale= tf.math.softplus(params[...,7:]))


inputs = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16,p17,p18, p19]).transpose()
outputs = np.array([fitx_10, fitx_11, fitx_12, fitx_13, fitx_14, fitx_15, tex]).transpose()/100


x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=47)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=22)


inputs = Input(shape=(19,))
hidden = Dense(100, activation='tanh')(inputs)
hidden = Dense(80, activation= 'tanh')(hidden)
hidden = Dense(80, activation= 'tanh')(hidden)
hidden = Dense(80, activation= 'tanh')(hidden)
hidden = Dense(80, activation= 'tanh')(hidden)
hidden = Dense(80, activation= 'tanh')(hidden)
hidden = Dense(50, activation= 'tanh')(hidden)
hidden = Dense(50, activation= 'tanh')(hidden)
hidden = Dense(50, activation= 'tanh')(hidden)
hidden = Dense(50, activation= 'tanh')(hidden)
hidden = Dense(50, activation= 'tanh')(hidden)
hidden = Dense(50, activation= 'tanh')(hidden)
hidden = Dense(30, activation= 'tanh')(hidden)
hidden = Dense(20, activation= 'tanh')(hidden)

params = Dense(14)(hidden)
dist = tfp.layers.DistributionLambda(my_dist)(params)
model_sd_mean = Model(inputs=inputs, outputs=dist.mean())
model_sd_sd = Model(inputs=inputs, outputs=dist.stddev())

# total number of epochs
epochs = 20000
# Learning Rate Tuning 
initial_learning_rate = 1.0e-3
end_learning_rate= 0
power = 0.75
decay_rate = 0.96
# 10k epochs got poly: ok

# polynomial learning rate
lr_poly = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate= initial_learning_rate,
    decay_steps = 4.15*epochs,
    end_learning_rate= end_learning_rate,
    power = power)

# exponetial learning rate
lr_exp = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate= initial_learning_rate,
    decay_steps = epochs,
    decay_rate = decay_rate)

model = Model(inputs=inputs, outputs=dist)
model.compile(Adam(learning_rate=lr_poly), loss= nll, metrics= ['mse'])

'''
history = model.fit(x_train, y_train, epochs=epochs, verbose=1, 
                       validation_data=(x_val,y_val))

plt.figure(figsize=(10,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('NLL')
plt.xlabel('Epochs')
#plt.yscale('log')
#plt.title('')
#plt.ylim(0,10)
#plt.show()


model.save("aleatoric_model_macho.h5")
model.save_weights('aleatoric_macho.h5')
# load model

'''





model.load_weights('aleatoric_macho.h5')
#model.summary()
# summarize model
#model.summary()
# evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print(score)
print("%s: %.8f%%" % (model.metrics_names[1], score[1]))
u=9
#print(outputs.reshape(7,1267)[:,1]*100)
x_pred = x_test[u:u+1,:]

print(x_pred)

runs = 20
nobay = np.zeros((runs,7))

for i in tqdm(range(0,runs)):
    nobay[i,:] = np.reshape(model.predict(x_pred)*100,7)
    
preds = np.mean(nobay, axis=0)  
preds_up = np.quantile(nobay, 0.975, axis=0)
preds_low = np.quantile(nobay, 0.025, axis=0)
#preds_up = np.quantile(nobay, 0.84, axis=0)
#preds_low = np.quantile(nobay, 0.16, axis=0)
true = ((y_test[u,:]))*100

def decoder(a,b):
    return (a*(np.max(b) - np.min(b))) + np.min(b)


preds_mean = np.array([decoder(preds[0], fit_10),decoder(preds[1], fit_11),decoder(preds[2], fit_12),
                         decoder(preds[3], fit_13),decoder(preds[4], fit_14),decoder(preds[5], fit_15),
                         decoder(preds[6], te)])

preds_ups = np.array([decoder(preds_up[0], fit_10),decoder(preds_up[1], fit_11),decoder(preds_up[2], fit_12),
                         decoder(preds_up[3], fit_13),decoder(preds_up[4], fit_14),decoder(preds_up[5], fit_15),
                         decoder(preds_up[6], te)])


preds_down = np.array([decoder(preds_low[0], fit_10),decoder(preds_low[1], fit_11),decoder(preds_low[2], fit_12),
                         decoder(preds_low[3], fit_13),decoder(preds_low[4], fit_14),decoder(preds_low[5], fit_15),
                         decoder(preds_low[6], te)])


true_mean = np.array([decoder(true[0], fit_10),decoder(true[1], fit_11),decoder(true[2], fit_12),
                         decoder(true[3], fit_13),decoder(true[4], fit_14),decoder(true[5], fit_15),
                         decoder(true[6], te)])


print(preds_mean)
print(true_mean)

E = np.linspace(1,22,10000)

def funcc(x, a,b,c):
    it1 = (c**-1.5)*a*np.exp(-(x/c)**b)
    return (it1*np.sqrt(x))

def unif_mean(x):
    result = list()
    for i in x:
        if i <= 5:
            result.append(funcc(i,preds_mean[0],preds_mean[1],preds_mean[2]))
        else:
            result.append(funcc(i,preds_mean[3],preds_mean[4],preds_mean[5]))
    return result

def unif_up(x):
    result = list()
    for i in x:
        if i <= 5:
            result.append(funcc(i,preds_down[0],preds_down[1],preds_down[2]))
        else:
            result.append(funcc(i,preds_down[3],preds_down[4],preds_down[5]))
    return result

def unif_down(x):
    result = list()
    for i in x:
        if i <= 5:
            result.append(funcc(i,preds_ups[0],preds_ups[1],preds_ups[2]))
        else:
            result.append(funcc(i,preds_ups[3],preds_ups[4],preds_ups[5]))
    return result

def unif_t(x):
    result = list()
    for i in x:
        if i <= 5:
            result.append(funcc(i,true_mean[0],true_mean[1],true_mean[2]))
        else:
            result.append(funcc(i,true_mean[3],true_mean[4],true_mean[5]))
    return result

def sav_unif(x):
    return sp.savgol_filter(x, 1051,2)

def sav_unif_t(x):
    return sp.savgol_filter(x, 451,2)


plt.figure(figsize=(8,8))
plt.title("Testing Data", fontsize=16, fontweight=16)
plt.plot(E, sav_unif(unif_mean(E)), 'r', linewidth=2)
plt.plot(E, (unif_up(E)), 'b--', linewidth=1)
plt.plot(E, (unif_down(E)), 'b--', linewidth=1)


plt.plot(E, sav_unif_t(unif_t(E)), 'g', linewidth=2)


plt.legend(['Predicted $f_p(E)$', '97.5% prec.', '2.50% prec.', 'True $f_p(E)$'])
plt.yscale('log')
plt.xlabel('Energy (eV)', fontsize=11)
plt.ylabel('$f_p(E)$ $(eV^{-3/2})$', fontsize=11)
plt.grid(True)
fig = plt.gcf()
plt.ylim(1e-3,2*2e-1)
plt.xlim(0,20)
plt.tick_params(direction='in')
plt.tick_params(which='minor', direction='in')
#plt.fill_between(E, unif_down(E),unif_up(E), fc='lightskyblue', ec='None', alpha=0.7)
print('Electron Temperatures (eV)', preds_mean[-1], true_mean[-1])
#fig.savefig('Aleatoric_EEPF_ml.png', format='png', dpi=600)





