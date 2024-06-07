import pandas as pd
import numpy as np
import os
import glob
import argparse

parser = argparse.ArgumentParser(
    description=("Prepare LHCO dataset."),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--S_over_B", type=float, default=-1,
                    help="Signal over background ratio in the signal region.")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed for the mixing")
args = parser.parse_args()

# the "data" containing too much signal
features=pd.read_hdf("LCHO_RD_dataset.h5")

# additionally produced bkg
features_extrabkg = pd.read_hdf("produced_QCD_background.h5")

## to be split among the different sets 
features_extrabkg1 = features_extrabkg[:312858]

## to be used to enhance the evalaution
features_extrabkg2 = features_extrabkg[312858:]

features_sig=features[features['label']==1]
features_bg=features[features['label']==0]

# Read from data
mj1mj2_bg = np.array(features_bg[['mj1','mj2']])
tau21_bg = np.array(features_bg[['tau2j1','tau2j2']])/(1e-5+np.array(features_bg[['tau1j1','tau1j2']]))
mj1mj2_sig = np.array(features_sig[['mj1','mj2']])
tau21_sig = np.array(features_sig[['tau2j1','tau2j2']])/(1e-5+np.array(features_sig[['tau1j1','tau1j2']]))
mj1mj2_extrabkg1 = np.array(features_extrabkg1[['mj1','mj2']])
tau21_extrabkg1 = np.array(features_extrabkg1[['tau2j1','tau2j2']])/(1e-5+np.array(features_extrabkg1[['tau1j1','tau1j2']]))
mj1mj2_extrabkg2 = np.array(features_extrabkg2[['mj1','mj2']])
tau21_extrabkg2 = np.array(features_extrabkg2[['tau2j1','tau2j2']])/(1e-5+np.array(features_extrabkg2[['tau1j1','tau1j2']]))


# Sorting of mj1 and mj2:
# Identifies which column has the minimum of mj1 and mj2, and sorts it so the new array mjmin contains the 
# mj with the smallest energy, and mjmax is the one with the biggest.
mjmin_bg = mj1mj2_bg[range(len(mj1mj2_bg)), np.argmin(mj1mj2_bg, axis=1)] 
mjmax_bg = mj1mj2_bg[range(len(mj1mj2_bg)), np.argmax(mj1mj2_bg, axis=1)]
mjmin_sig = mj1mj2_sig[range(len(mj1mj2_sig)), np.argmin(mj1mj2_sig, axis=1)]
mjmax_sig = mj1mj2_sig[range(len(mj1mj2_sig)), np.argmax(mj1mj2_sig, axis=1)]
mjmin_extrabkg1 = mj1mj2_extrabkg1[range(len(mj1mj2_extrabkg1)), np.argmin(mj1mj2_extrabkg1, axis=1)] 
mjmax_extrabkg1 = mj1mj2_extrabkg1[range(len(mj1mj2_extrabkg1)), np.argmax(mj1mj2_extrabkg1, axis=1)]
mjmin_extrabkg2 = mj1mj2_extrabkg2[range(len(mj1mj2_extrabkg2)), np.argmin(mj1mj2_extrabkg2, axis=1)] 
mjmax_extrabkg2 = mj1mj2_extrabkg2[range(len(mj1mj2_extrabkg2)), np.argmax(mj1mj2_extrabkg2, axis=1)]

# Then we do the same sorting for the taus
tau21min_bg=tau21_bg[range(len(mj1mj2_bg)), np.argmin(mj1mj2_bg, axis=1)]
tau21max_bg=tau21_bg[range(len(mj1mj2_bg)), np.argmax(mj1mj2_bg, axis=1)]
tau21min_sig=tau21_sig[range(len(mj1mj2_sig)), np.argmin(mj1mj2_sig, axis=1)]
tau21max_sig=tau21_sig[range(len(mj1mj2_sig)), np.argmax(mj1mj2_sig, axis=1)]
tau21min_extrabkg1 = tau21_extrabkg1[range(len(mj1mj2_extrabkg1)), np.argmin(mj1mj2_extrabkg1, axis=1)]
tau21max_extrabkg1 = tau21_extrabkg1[range(len(mj1mj2_extrabkg1)), np.argmax(mj1mj2_extrabkg1, axis=1)]
tau21min_extrabkg2 = tau21_extrabkg2[range(len(mj1mj2_extrabkg2)), np.argmin(mj1mj2_extrabkg2, axis=1)]
tau21max_extrabkg2 = tau21_extrabkg2[range(len(mj1mj2_extrabkg2)), np.argmax(mj1mj2_extrabkg2, axis=1)]


# Calculate mjj and collect the features into a dataset, plus mark signal/bg with 1/0
pjj_sig = (np.array(features_sig[['pxj1','pyj1','pzj1']])+np.array(features_sig[['pxj2','pyj2','pzj2']]))
Ejj_sig = np.sqrt(np.sum(np.array(features_sig[['pxj1','pyj1','pzj1','mj1']])**2, axis=1))\
    +np.sqrt(np.sum(np.array(features_sig[['pxj2','pyj2','pzj2','mj2']])**2, axis=1))
mjj_sig = np.sqrt(Ejj_sig**2-np.sum(pjj_sig**2, axis=1))

pjj_bg = (np.array(features_bg[['pxj1','pyj1','pzj1']])+np.array(features_bg[['pxj2','pyj2','pzj2']]))
Ejj_bg = np.sqrt(np.sum(np.array(features_bg[['pxj1','pyj1','pzj1','mj1']])**2, axis=1))\
    +np.sqrt(np.sum(np.array(features_bg[['pxj2','pyj2','pzj2','mj2']])**2, axis=1))
mjj_bg = np.sqrt(Ejj_bg**2-np.sum(pjj_bg**2, axis=1))

pjj_extrabkg1 = (np.array(features_extrabkg1[['pxj1','pyj1','pzj1']])+np.array(features_extrabkg1[['pxj2','pyj2','pzj2']]))
Ejj_extrabkg1 = np.sqrt(np.sum(np.array(features_extrabkg1[['pxj1','pyj1','pzj1','mj1']])**2, axis=1))\
    +np.sqrt(np.sum(np.array(features_extrabkg1[['pxj2','pyj2','pzj2','mj2']])**2, axis=1))
mjj_extrabkg1 = np.sqrt(Ejj_extrabkg1**2-np.sum(pjj_extrabkg1**2, axis=1))
pjj_extrabkg2 = (np.array(features_extrabkg2[['pxj1','pyj1','pzj1']])+np.array(features_extrabkg2[['pxj2','pyj2','pzj2']]))
Ejj_extrabkg2 = np.sqrt(np.sum(np.array(features_extrabkg2[['pxj1','pyj1','pzj1','mj1']])**2, axis=1))\
    +np.sqrt(np.sum(np.array(features_extrabkg2[['pxj2','pyj2','pzj2','mj2']])**2, axis=1))
mjj_extrabkg2 = np.sqrt(Ejj_extrabkg2**2-np.sum(pjj_extrabkg2**2, axis=1))


# LHCO
dataset_bg = np.dstack((mjj_bg/1000, mjmin_bg/1000, (mjmax_bg-mjmin_bg)/1000, tau21min_bg, tau21max_bg, np.zeros(len(mjj_bg))))[0]
dataset_sig = np.dstack((mjj_sig/1000, mjmin_sig/1000, (mjmax_sig-mjmin_sig)/1000, tau21min_sig, tau21max_sig, np.ones(len(mjj_sig))))[0]

LHCO_full = np.concatenate((dataset_bg, dataset_sig))
pd.DataFrame(LHCO_full).to_csv("full_LHCO_RnD.csv", index=False)


# Simulation
dataset_extrabkg1 = np.dstack((mjj_extrabkg1/1000, mjmin_extrabkg1/1000, (mjmax_extrabkg1-mjmin_extrabkg1)/1000, tau21min_extrabkg1, tau21max_extrabkg1, np.zeros(len(mjj_extrabkg1))))[0]
dataset_extrabkg2 = np.dstack((mjj_extrabkg2/1000, mjmin_extrabkg2/1000, (mjmax_extrabkg2-mjmin_extrabkg2)/1000, tau21min_extrabkg2, tau21max_extrabkg2, np.zeros(len(mjj_extrabkg2))))[0]

simulation_full = np.concatenate((dataset_extrabkg1, dataset_extrabkg2))
pd.DataFrame(simulation_full).to_csv("full_simulation.csv", index=False)


np.random.seed(args.seed) # Set the random seed so we get a deterministic result

if args.seed != 1:
    np.random.shuffle(dataset_sig)

if args.S_over_B == -1:
    n_sig = 1000
else:
    n_sig = int(args.S_over_B * 1000 / 0.006361658645922605)

data_all = np.concatenate((dataset_bg, dataset_sig[:n_sig]))
indices = np.array(range(len(data_all))).astype('int')
np.random.shuffle(indices)
data_all = data_all[indices]

indices_extrabkg1 = np.array(range(len(dataset_extrabkg1))).astype('int')
np.random.shuffle(indices_extrabkg1)
dataset_extrabkg1 = dataset_extrabkg1[indices_extrabkg1]
indices_extrabkg2 = np.array(range(len(dataset_extrabkg2))).astype('int')
np.random.shuffle(indices_extrabkg2)
dataset_extrabkg2 = dataset_extrabkg2[indices_extrabkg2]

# format of data_all: mjj (TeV), mjmin (TeV), mjmax-mjmin (TeV), tau21(mjmin), tau21 (mjmax), sigorbg label

minmass=3.3
maxmass=3.7

innermask = (data_all[:,0]>minmass) & (data_all[:,0]<maxmass)
outermask = ~innermask
innerdata = data_all[innermask]
outerdata = data_all[outermask]

innermask_extrabkg1 = (dataset_extrabkg1[:,0]>minmass) & (dataset_extrabkg1[:,0]<maxmass)
innerdata_extrabkg1 = dataset_extrabkg1[innermask_extrabkg1]
innermask_extrabkg2 = (dataset_extrabkg2[:,0]>minmass) & (dataset_extrabkg2[:,0]<maxmass)
innerdata_extrabkg2 = dataset_extrabkg2[innermask_extrabkg2]


outerdata_train = outerdata[:500000]
outerdata_val = outerdata[500000:]

innerdata_train = innerdata[:60000]
innerdata_val = innerdata[60000:120000]

innerdata_extrasig = dataset_sig[n_sig:]
innerdata_extrasig = innerdata_extrasig[(innerdata_extrasig[:,0]>minmass) & (innerdata_extrasig[:,0]<maxmass)]

## splitting extra signal into train, val and test set
n_sig_test = 20000
n_extrasig_train =  (innerdata_extrasig.shape[0]-n_sig_test)//2
innerdata_extrasig_test = innerdata_extrasig[:n_sig_test]
innerdata_extrasig_train = innerdata_extrasig[n_sig_test:n_sig_test+n_extrasig_train]
innerdata_extrasig_val = innerdata_extrasig[n_sig_test+n_extrasig_train:]

## splitting extra bkg (1) into train, val and test set
n_bkg_test = 40000
n_extrabkg_train =  (innerdata_extrabkg1.shape[0]-n_bkg_test)//2
innerdata_extrabkg1_test = innerdata_extrabkg1[:n_bkg_test]
innerdata_extrabkg1_train = innerdata_extrabkg1[n_bkg_test:n_bkg_test+n_extrabkg_train]
innerdata_extrabkg1_val = innerdata_extrabkg1[n_bkg_test+n_extrabkg_train:]

## putting together artificial test set
innerdata_test = np.vstack((innerdata_extrabkg1_test, innerdata_extrasig_test))

# np.save(os.path.join(args.outdir, 'outerdata_train.npy'), outerdata_train)
# np.save(os.path.join(args.outdir, 'outerdata_test.npy'), outerdata_val)
# np.save(os.path.join(args.outdir, 'innerdata_train.npy'), innerdata_train)
# np.save(os.path.join(args.outdir, 'innerdata_val.npy'), innerdata_val)   
# np.save(os.path.join(args.outdir, 'innerdata_test.npy'), innerdata_test)      
# np.save(os.path.join(args.outdir, 'innerdata_extrasig_train.npy'), innerdata_extrasig_train)
# np.save(os.path.join(args.outdir, 'innerdata_extrasig_val.npy'), innerdata_extrasig_val)
# np.save(os.path.join(args.outdir, 'innerdata_extrabkg_train.npy'), innerdata_extrabkg1_train)
# np.save(os.path.join(args.outdir, 'innerdata_extrabkg_val.npy'), innerdata_extrabkg1_val)
# np.save(os.path.join(args.outdir, 'innerdata_extrabkg_test.npy'), innerdata_extrabkg2)

innerdata_extrabkg_train = innerdata_extrabkg1_train
innerdata_extrabkg_val = innerdata_extrabkg1_val
innerdata_extrabkg_test = innerdata_extrabkg2

# for d in ['LHCO_dataset', 'produced_background']:
#     try:
#         os.makedirs(d)
#     finally:
#         pass

# pd.DataFrame(outerdata_train).to_csv(os.path.join("LHCO_dataset",                 "SB_train.csv"))
# pd.DataFrame(outerdata_val).to_csv(os.path.join("LHCO_dataset",                   "SB_val.csv"))
# pd.DataFrame(innerdata_train).to_csv(os.path.join("LHCO_dataset",                 "SR_train.csv"))
# pd.DataFrame(innerdata_val).to_csv(os.path.join("LHCO_dataset",                   "SR_val.csv"))
# pd.DataFrame(innerdata_test).to_csv(os.path.join("LHCO_dataset",                  "SR_test.csv"))
# pd.DataFrame(innerdata_extrasig_train).to_csv(os.path.join("LHCO_dataset",        "extrasig_SR_train.csv"))
# pd.DataFrame(innerdata_extrasig_val).to_csv(os.path.join("LHCO_dataset",          "extrasig_SR_val.csv"))
# pd.DataFrame(innerdata_extrasig_test).to_csv(os.path.join("LHCO_dataset",         "extrasig_SR_test.csv"))

# pd.DataFrame(innerdata_extrabkg_train).to_csv(os.path.join("produced_background", "extrabg_SR_train.csv"))
# pd.DataFrame(innerdata_extrabkg_val).to_csv(os.path.join("produced_background",   "extrabg_SR_val.csv"))
# pd.DataFrame(innerdata_extrabkg_test).to_csv(os.path.join("produced_background",  "extrabg_SR_test.csv"))

for d in ['mock', 'simulation', 'evaluation']:
    try:
        os.makedirs(d)
    finally:
        pass


SB_train = pd.DataFrame(outerdata_train)
SB_val =   pd.DataFrame(outerdata_val)
SR_train = pd.DataFrame(innerdata_train)
SR_val =   pd.DataFrame(innerdata_val)

SB_train.to_csv(os.path.join("mock", "SB_train.csv"), index=False)
SB_val.to_csv(os.path.join("mock",   "SB_val.csv"), index=False)
SR_train.to_csv(os.path.join("mock", "SR_train.csv"), index=False)
SR_val.to_csv(os.path.join("mock",   "SR_val.csv"), index=False)

extrasig_SR_train = pd.DataFrame(innerdata_extrasig_train)
extrasig_SR_val =   pd.DataFrame(innerdata_extrasig_val)
extrabg_SR_train =  pd.DataFrame(innerdata_extrabkg_train)
extrabg_SR_val =    pd.DataFrame(innerdata_extrabkg_val)

extrasig_SR_train.to_csv(os.path.join("simulation", "extrasig_SR_train.csv"), index=False)
extrasig_SR_val.to_csv(os.path.join("simulation",   "extrasig_SR_val.csv"), index=False)
extrabg_SR_train.to_csv(os.path.join("simulation",  "extrabg_SR_train.csv"), index=False)
extrabg_SR_val.to_csv(os.path.join("simulation",    "extrabg_SR_val.csv"), index=False)

SR_test =         pd.DataFrame(innerdata_test)
extrabg_SR_test = pd.DataFrame(innerdata_extrabkg_test)

SR_test.to_csv(os.path.join("evaluation",           "SR_test.csv"), index=False)
extrabg_SR_test.to_csv(os.path.join("evaluation",   "extrabg_SR_test.csv"), index=False)

eval_list = tuple([pd.read_csv(csv) for csv in glob.glob('evaluation/*.csv')])
evaluation = pd.concat(eval_list)
evaluation.to_csv("evaluation_set.csv", index=False)

mock_list = tuple([pd.read_csv(csv) for csv in glob.glob('mock/*.csv')])
mock = pd.concat(mock_list)
mock.to_csv("mock_set.csv", index=False)

sim_list = tuple([pd.read_csv(csv) for csv in glob.glob('simulation/*.csv')])
simulation = pd.concat(sim_list)
simulation.to_csv("simulation_set.csv", index=False)
