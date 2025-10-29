import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from netneurotools import datasets, plotting
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import LinearRegression
from nilearn.datasets import fetch_atlas_schaefer_2018
from sklearn.decomposition import PCA
meandata = np.genfromtxt('J:\\MCI_nii\\ASL_BOLD\\mean.txt', delimiter=',')
receptor = np.genfromtxt('E:\\software\\hansen_receptors-main\\results\\receptor_data_scale400.csv', delimiter=',')


path = 'E:/software/hansen_receptors-main/'
scale = 'scale16'

receptors_csv = [path+'data/PET_parcellated/'+scale+'/5HT1a_way_hc36_savli.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT1b_p943_hc22_savli.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT1b_p943_hc65_gallezot.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT2a_cimbi_hc29_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT4_sb20_hc59_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/5HT6_gsk_hc30_radhakrishnan.csv',
                 path+'data/PET_parcellated/'+scale+'/5HTT_dasb_hc100_beliveau.csv',
                 path+'data/PET_parcellated/'+scale+'/A4B2_flubatine_hc30_hillmer.csv',
                 path+'data/PET_parcellated/'+scale+'/CB1_omar_hc77_normandin.csv',
                 path+'data/PET_parcellated/'+scale+'/D1_SCH23390_hc13_kaller.csv',
                 path+'data/PET_parcellated/'+scale+'/D2_flb457_hc37_smith.csv',
                 path+'data/PET_parcellated/'+scale+'/D2_flb457_hc55_sandiego.csv',
                 path+'data/PET_parcellated/'+scale+'/DAT_fpcit_hc174_dukart_spect.csv',
                 path+'data/PET_parcellated/'+scale+'/GABAa-bz_flumazenil_hc16_norgaard.csv',
                 path+'data/PET_parcellated/'+scale+'/H3_cban_hc8_gallezot.csv', 
                 path+'data/PET_parcellated/'+scale+'/M1_lsn_hc24_naganawa.csv',
                 path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc22_rosaneto.csv',
                 path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc28_dubois.csv',
                 path+'data/PET_parcellated/'+scale+'/mGluR5_abp_hc73_smart.csv',
                 path+'data/PET_parcellated/'+scale+'/MU_carfentanil_hc204_kantonen.csv',
                 path+'data/PET_parcellated/'+scale+'/NAT_MRB_hc77_ding.csv',
                 path+'data/PET_parcellated/'+scale+'/NMDA_ge179_hc29_galovic.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc4_tuominen.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc5_bedard_sum.csv',
                 path+'data/PET_parcellated/'+scale+'/VAChT_feobv_hc18_aghourian_sum.csv']

# combine all the receptors (including repeats)
r = np.zeros([nodes, len(receptors_csv)])
for i in range(len(receptors_csv)):
    r[:, i] = np.genfromtxt(receptors_csv[i], delimiter=',')

receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "NMDA", "VAChT"])
np.save(path+'data/receptor_names_pet.npy', receptor_names)

# make final region x receptor matrix

receptor_data = np.zeros([nnodes, len(receptor_names)])
receptor_data[:, 0] = r[:, 0]
receptor_data[:, 2:9] = r[:, 3:10]
receptor_data[:, 10:14] = r[:, 12:16]
receptor_data[:, 15:18] = r[:, 19:22]

# weighted average of 5HT1B p943
receptor_data[:, 1] = (zscore(r[:, 1])*22 + zscore(r[:, 2])*65) / (22+65)

# weighted average of D2 flb457
receptor_data[:, 9] = (zscore(r[:, 10])*37 + zscore(r[:, 11])*55) / (37+55)

# weighted average of mGluR5 ABP688
receptor_data[:, 14] = (zscore(r[:, 16])*22 + zscore(r[:, 17])*28 + zscore(r[:, 18])*73) / (22+28+73)

# weighted average of VAChT FEOBV
receptor_data[:, 18] = (zscore(r[:, 22])*3 + zscore(r[:, 23])*4 + zscore(r[:, 24]) + zscore(r[:, 25])) / \
                       (3+4+5+18)

np.savetxt(path+'results/receptor_data_'+scale+'.csv', receptor_data, delimiter=',')


import numpy as np
import matplotlib.pyplot as plt
from netneurotools import stats
from scipy.stats import zscore, pearsonr, ttest_ind
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import LinearRegression
from nilearn.datasets import fetch_atlas_schaefer_2018
from statsmodels.stats.multitest import multipletests
def get_reg_r_sq(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * \
        (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return adjusted_r_squared


def get_reg_r_pval(X, y, spins, nspins):
    emp = get_reg_r_sq(X, y)
    null = np.zeros((nspins, ))
    for s in range(nspins):
        null[s] = get_reg_r_sq(X[spins[:, s], :], y)
    return (1 + sum(null > emp))/(nspins + 1)



receptor = np.genfromtxt('E:\\software\\hansen_receptors-main\\results\\receptor_data_scale400new.csv', delimiter=',')
meandata = np.genfromtxt('F:\\MCI_all\\ASL_BOLD\\meanscore.txt', delimiter=',')
meandata = np.genfromtxt('F:\\MCI_all\\ASL_BOLD\\meanscore.txt', delimiter=',')

nnodes=416
path='E:/software/hansen_receptors-main'
hemiid = np.zeros((nnodes, ))
hemiid[:int(nnodes/2)] = 1
coords = np.genfromtxt(path+'/data/schaefer/coordinates/Schaefer_416_centers.txt')[:, 1:]
nspins = 5000



nnodes=398
hemiid = np.zeros((nnodes, ))
hemiid[:int(nnodes/2)] = 1
coords = np.genfromtxt(path+'/data/schaefer/coordinates/Schaefer_398_centers.txt')[:, 1:]
nspins = 1000

spins = stats.gen_spinsamples(coords, hemiid, n_rotate=nspins, seed=1234)





path='E:/software/hansen_receptors-main'

model_pval= get_reg_r_pval(receptor_data,zscore(power[:, i]), spins, nspins)

for i in range(416):
    model_pval= get_reg_r_pval(receptor_data,data[i,:].T, spins, nspins)
    P_values[i] = model_pval



from netneurotools.stats import get_dominance_stats
model_metrics, model_r_sq = get_dominance_stats(receptor, meandata)
model_metrics, model_r_sq = get_dominance_stats(receptor, meandata)
sns.heatmap(data=model_metrics['individual_dominance'],xticklabels=["delta","theta","alpha1","alpha2","beta1","beta2","beta3","gamma"],vmax=0.25)

sns.heatmap(data=model_metrics['partial_dominance'],xticklabels=["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2","CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5","MOR", "NET", "NMDA", "VAChT"], yticklabels=["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2","CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5","MOR", "NET", "NMDA"])



sns.heatmap(data=model_metrics['individual_dominance'],xticklabels=["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2","CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5","MOR", "NET", "NMDA", "VAChT"])