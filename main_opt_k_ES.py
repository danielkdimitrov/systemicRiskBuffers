# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:58:29 2023

@author: danie
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from DataLoad import DataTransform, DataLoad
from setParams import SetParams
from optimalSystemicCapital import PDmodel
from getECost import getECost
from myplotstyle import * 
#from GetSystemicRiskSims import * # ??
from GetImpliedParams import *

from datetime import timedelta

from scipy.stats import norm
#from scipy.optimize import root, minimize, Bounds
from statsmodels.stats.correlation_tools import cov_nearest
import matplotlib.patches as patches

import pickle

path = r"C:\\Users\\danie\\Dropbox\\Amsterdam\\Systemic Risk Europe DNB\\imagesTemp\\"


'''
Optimize over k_bar  -- European Universe
'''


#%%% DATA DRIVEN. Load Full Data
mySRparams = GetImpliedParams(fixedRRs = True)
mySRparams.DataSet.banks.loc['DANK', 'p2r  CET1'] = mySRparams.DataSet.banks['p2r  CET1'].median()

paramsDict = {}

paramsDict['Sigma'] = mySRparams.dfSigmaEval.values.astype(np.float64) 
paramsDict['wts'] = mySRparams.wtsEval.values.astype(np.float64) 
paramsDict['LGD'] = mySRparams.LGDEval.values.astype(np.float64)
paramsDict['Rho'] = mySRparams.ldngs.astype(np.float64)
paramsDict['Names'] = mySRparams.dfSigmaEval.columns

paramsDict['O-SII rates'] = mySRparams.DataSet.banks.loc[mySRparams.dfCDS.columns,['O-SII buffer']].values.T[0]
paramsDict['k_bar'] = np.sum(paramsDict['O-SII rates']*paramsDict['wts']) #  #
paramsDict['Lbar'] = .0

'including pillar 2:'
paramsDict['k_p2r'] = mySRparams.DataSet.banks['p2r  CET1'][mySRparams.universe].values.astype(np.float64)



#%% --------------- K-BAR RRELATIVE TO O-SI AVERAGE ---------------------
#%%---------------- OPTIMIZE RELATIVE TO EURO SCALE  ---------------------
#%%% Minimize ES at current  
'Optimize vs. kbar = O-SII Rate, EURO-wide'

myPD = PDmodel('min ES', paramsDict, True)

#%%%
dfTable = pd.DataFrame(index =paramsDict['Names'], columns = ['Country','Names','k','k_osii','k_bar_loc','k_bar','k_macro_str_osii_eur','k_macro_str_osii_local'])
dfTable['k_macro_str_osii_eur'] = myPD.dict['k_macro_str']*100 
dfTable['k_bar'] = paramsDict['k_bar']*100 


dfTable['Names'] = paramsDict['Names']
dfTable['Country'] =  mySRparams.DataSet.banks.Country
dfTable['k'] = mySRparams.DataSet.capitalRatio.iloc[0]
dfTable['k_osii'] =  paramsDict['O-SII rates']*100

#%%---------------- 2. OPTIMIZE RELATIVE TO LOCAL SCALE  ---------------------
#%%
'Optimize vs. kbar = O-SII Rate, at country level'
countries = ['Netherlands', 'Germany', 'Spain','France', 'Sweden', 'Italy']

for jc, country in enumerate(countries):
    
    mask = dfTable['Country'] == country
    print(country)
    
    paramsDict1 = {}

    paramsDict1['Sigma'] = mySRparams.dfSigmaEval.T[mask].values.astype(np.float64).T
    paramsDict1['wts'] = mySRparams.wtsEval[mask].values.astype(np.float64) / np.sum(mySRparams.wtsEval[mask].values.astype(np.float64))
    paramsDict1['LGD'] =  mySRparams.LGDEval[mask].values.astype(np.float64)
    paramsDict1['Rho'] = mySRparams.ldngs[mask].astype(np.float64)
    paramsDict1['Names'] = mySRparams.dfSigmaEval.T[mask].index        
    paramsDict1['O-SII rates'] = mySRparams.DataSet.banks.loc[mySRparams.dfCDS.columns,['O-SII buffer']][mask].values.T[0]
    paramsDict1['k_bar'] = np.sum(paramsDict1['O-SII rates']*paramsDict1['wts'])
    paramsDict1['Lbar'] = .0
    
    'including pillar 2:'
    paramsDict1['k_p2r'] = mySRparams.DataSet.banks.loc[mySRparams.dfCDS.columns,'p2r  CET1'][mask].values.astype(np.float64)
    
    myPD1 = PDmodel('min ES', paramsDict1, True)
    
    dfTable.loc[mask,'k_macro_str_osii_local'] = myPD1.dict['k_macro_str']*100
    dfTable.loc[mask,'k_bar_loc'] = paramsDict1['k_bar']*100
    
#%% plot k vs k_str, Figure "Optimal Macro Buffers imposing the current O-SII average"
dfTable.sort_values(['Country','Names'],  ascending = [True, True], inplace = True)
dfTable.to_excel(r'images\OSIIrateOpt.xlsx')


#%%---------------- PLOT O-SII CHART ---------------------

'Plot Local Scale'
dfTable_loc = dfTable.iloc[4:, [0,3,7]]
hatch_patterns = ['', '//', '.', '-', '']
color_patterns = ['black','white','grey']


plot2comparison(df_plot, hatch_patterns, color_patterns)
saveFig(path,'OsiiModelRank_loc') 


'Plot Euro Scales'
dfTable_loc = dfTable.iloc[:, [0,3,6]]

#pd.read_csv('OsiiEeiESS.csv', index_col='Short Code')
#dfTable.drop('EEI', axis=1, inplace = True)

#dfTable = pd.read_csv('OsiiESS.csv', index_col='Short Code')


#color_patterns = ['grey','g','tab:orange']

#hatch_patterns = ['.', '//.', '', '-', '']
#color_patterns = ['grey','white','black']



#saveFig(path,'OsiiModelRank_eur') 




























# %%
df = pd.DataFrame(index =paramsDict['Names'], columns = ['O-SII Rate', 'k_macro_str'])
#df['k_macro_str_osii'] = myPD.dict['k_macro_str']*100 
#df['O-SII Rate'] =  paramsDict['O-SII rates']*100
#dfTable['k_macro_str_osii'] = df['k_macro_str_osii']
#dfTable['O-SII Rate'] = df['O-SII Rate']

hatch_patterns = ['', '.', '//', '-', '']
#color_patterns = ['grey','g','tab:orange']
color_patterns = ['black','grey','white']

'plot'
ax = myplot_frame((16, 6))
dfTable[['k_osii','k_macro_str_osii_local','k_macro_str_osii_eur']].plot.bar(align='center',color= color_patterns, edgecolor='black',alpha=.75, ax = ax)
plt.ylabel(r'$k_{i,macro}(\%)$', fontsize = 18)

'Apply hatch patterns based on bar color'
for i, (col, color) in enumerate(zip(dfTable[['k_osii','k_macro_str_osii_local','k_macro_str_osii_eur']].columns[1:], color_patterns)):
    for j, bar in enumerate(ax.patches[i * len(df):(i + 1) * len(df)]):
        if color == color_patterns[0]:
            bar.set_hatch(hatch_patterns[0])
        elif color == color_patterns[1]:
            bar.set_hatch(hatch_patterns[1])
        elif color == color_patterns[2]:
            bar.set_hatch(hatch_patterns[2])            

plt.xticks(fontsize=18,rotation=75)  # Adjust font size here
plt.yticks(fontsize=18)  # Adjust font size here
plt.gca().xaxis.set_tick_params(which='minor', size=0)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().tick_params(axis='x', which='minor', bottom='off')
plt.legend(['O-SII Rate','Model Optimal (Local)','Model Optimal (Euro-wide)'], fontsize = 14,loc='upper center', ncol=3,bbox_to_anchor=(0.5, -0.2))

# Add vertical lines between different categories
categories = dfTable['Country'].unique()

count = 0 
country_count = 'abs'
for idx, df1 in dfTable.iterrows():
    if df1.Country == country_count:
        continue
    country_count = df1.Country
    line_position = dfTable.index.get_loc(dfTable.index[dfTable.Country == country_count][-1]) + 0.5
    plt.axvline(line_position, color='gray', linestyle='--')
    # annotate
    plt.annotate(df1.Country[:2], (line_position- .5, 2.9), textcoords="offset points", xytext=(0,2.9), ha='center', fontsize=25)    

saveFig(path,'OsiiModelRank') 



#%%---------------- 3. RUN EEI DUTCH CHARTS  ---------------------
country = 'Netherlands'

mask = dfTable['Country'] == country

paramsDict1 = {}

paramsDict1['Sigma'] = mySRparams.dfSigmaEval.T[mask].values.astype(np.float64).T
paramsDict1['wts'] = mySRparams.wtsEval[mask].values.astype(np.float64) / np.sum(mySRparams.wtsEval[mask].values.astype(np.float64))
paramsDict1['LGD'] =  mySRparams.LGDEval[mask].values.astype(np.float64)
paramsDict1['Rho'] = mySRparams.ldngs[mask].astype(np.float64)
paramsDict1['Names'] = mySRparams.dfSigmaEval.T[mask].index        
paramsDict1['O-SII rates'] = mySRparams.DataSet.banks.loc[mySRparams.dfCDS.columns,['O-SII buffer']][mask].values.T[0]
paramsDict1['k_bar'] = np.sum(paramsDict1['O-SII rates']*paramsDict1['wts'])
paramsDict1['Lbar'] = .0

'including pillar 2:'
paramsDict1['k_p2r'] = mySRparams.DataSet.banks.loc[mySRparams.dfCDS.columns,'p2r  CET1'][mask].values.astype(np.float64)

myPD1 = PDmodel('min ES', paramsDict1, True)

'Sensitivity analysis'
K_bar = np.linspace(0.01, .15, 10)

dfES = pd.DataFrame(index = K_bar, columns = ['ES'])
dfMES = pd.DataFrame(index = K_bar, columns = paramsDict1['Names'])
dfKimacro = pd.DataFrame(index = K_bar, columns = paramsDict1['Names'])
dfPD = pd.DataFrame(index = K_bar, columns = paramsDict1['Names'])

for jK, k_bar in enumerate(K_bar):
    print(k_bar)
    paramsDict1['k_bar'] = k_bar
    myPD = PDmodel('min ES', paramsDict1, True)
    dfES.loc[k_bar] = myPD.dict['ESopt']
    dfMES.loc[k_bar] = myPD.dict['MESopt']
    dfKimacro.loc[k_bar] = myPD.dict['k_macro_str']
    dfPD.loc[k_bar] = myPD.dict['PD']

'Nice plots'
x = dfKimacro.index*100
lbls = dfKimacro.columns

'1: k_i'
yy = dfKimacro.to_numpy()*100
lbls = dfKimacro.columns
ax = myplotNLines(x,yy.T,r'$k_{i,macro}(\%)$',lbls,r'$\overline{k}(\%)$','%.2f')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.rc('legend',fontsize=15)
saveFig(path,'ES_Dutch_k') 
'2: ES'
yy = dfES.to_numpy()
ax = myplotNLines(x,yy.T,r'$ES$',lbls,r'$\overline{k}(\%)$','%.2f')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.rc('legend',fontsize=15)
saveFig(path,'ES_Dutch') 
'3: ES improvement'
yy = 100*(myPD.dict['ESmicro']- dfES.to_numpy())/myPD.dict['ESmicro']
ax = myplotNLines(x,yy.T,'ES Reduction (%)',lbls,r'$\overline{k}(\%)$','%.2f')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.rc('legend',fontsize=15)
saveFig(path,'ES_reduc_Dutch') 
'4: PD'
yy = dfPD.to_numpy()*100
ax = myplotNLines(x,yy.T,'PD (%)',lbls,r'$\overline{k}(\%)$','%.2f')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.rc('legend',fontsize=15)
saveFig(path,'PD_Dutch') 
'5: MES'
ax =myplot_frame((8, 6))
ax.set_xlabel(r'$\overline{k}(\%)$')
ax.set_ylabel(r'$ESS$, Weighted $MES(\%)$')

wdfMES.multiply(100).plot.area(ax=ax, cmap='Oranges')
# Get the current x-axis labels
xticks = ax.get_xticks()
# Multiply the x-axis labels by 100
xtick_labels = [str(int(label * 100)) for label in xticks]
# Set the modified x-axis labels]
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax.set_xticklabels(xtick_labels)
plt.legend(['ABN','INGB','RABO','VB'])
plt.rc('legend',fontsize=15)
saveFig(path,'MES_Dutch') 

#%%---------------- 4. RUN POLICYMAKER OBJECTIVE TO DETERMINE K-BAR ---------------------
#%%

paramsDict['Lbar'] = 0.09 #.4 
paramsDict['GDP Loss'] = .09
myDict_K_bar = getECost(paramsDict) #k_micro = 8.75 at Lbar = .4%%
Ecost, dfPDsys, dfES, SCB, k_bar_min = myDict_K_bar['ECost'], myDict_K_bar['dfPDsys'], myDict_K_bar['dfES'], myDict_K_bar['SCB'], myDict_K_bar['k_bar_min']

paramsDict['GDP Loss'] = .06
myDict_K_bar2 = getECost(paramsDict) #k_micro = 7.5 at Lbar = 0.0225%
Ecost2, dfPDsys2, dfES2, SCB3, k_bar_min2 = myDict_K_bar2['ECost'], myDict_K_bar2['dfPDsys'], myDict_K_bar2['dfES'], myDict_K_bar2['SCB'], myDict_K_bar2['k_bar_min']

paramsDict['GDP Loss'] = .04
myDict_K_bar3 = getECost(paramsDict) #k_micro = 7.5 at Lbar = 0.0225%
Ecost3, dfPDsys3, dfES3, SCB3, k_bar_min3 = myDict_K_bar3['ECost'], myDict_K_bar3['dfPDsys'], myDict_K_bar3['dfES'], myDict_K_bar3['SCB'], myDict_K_bar3['k_bar_min']


'''
# Remove observations with percentage change greater than the threshold
def myFilter(Ecost):
    Ecost_filter = Ecost.copy()
    percentage_change = Ecost_filter.pct_change()
    Ecost_filter[abs(percentage_change) > .2] = None
    for col in Ecost_filter:
        Ecost_filter[col] = pd.to_numeric(Ecost_filter[col], errors='coerce')
    # Interpolate missing values
    Ecost_filter.interpolate(method='linear', inplace=True)
    return Ecost_filter

Ecost_ = myFilter(Ecost)
Ecost2_ = myFilter(Ecost2)
Ecost3_ = myFilter(Ecost3)
'''

ax = myplotNLines(Ecost.index*100, [Ecost.values*100, Ecost2.values*100, Ecost3.values*100],'Welfare Cost',['Severe','Moderate', 'Milder'],r'$\overline{k} (\%)$','%.2f')  #[r'$к_{micro}=7pc$',r'$к_{micro}=8pc$'
ax.legend(fontsize=25)
#line_position = Ecost.index.get_loc(Ecost.index[dfTable.Country == country_count][-1]) + 0.5
plt.axvline(x = k_bar_min.index*100, color='gray', linestyle='-')
plt.axvline(x = k_bar_min2.index*100, color='gray', linestyle='--')
plt.axvline(x = k_bar_min3.index*100, color='gray', linestyle=':')
saveFig(path,'overallCast_varyKmicro')

ax = myplotNLines(Ecost.index*100, [dfPDsys.values*100, dfPDsys3.values*100],r'$P(\overline{k})(\%)$',[r'$P(L>40\%)$',r'$P(L>50\%)$'],r'$\overline{k} (\%)$','%.1f')  #[r'$к_{micro}=7pc$',r'$к_{micro}=8pc$'

ax = myplot(Ecost.index*100, dfPDsys.values*100, [r'$\overline{k} (\%)$', r'$P(\overline{k})(\%)$'],'%.1f')  #[r'$к_{micro}=7pc$',r'$к_{micro}=8pc$'
plt.axvline(x = k_bar_min.index*100, color='gray', linestyle='-')
plt.axvline(x = k_bar_min2.index*100, color='gray', linestyle='--')
plt.axvline(x = k_bar_min3.index*100, color='gray', linestyle=':')
ax.legend(fontsize=25)
saveFig(path,'PD_varyKmicro')

#pio.templates.default = "plotly"


#%% 

#%%%  min ES socially optimal k_bars or current k_bar !! 

# mySRparams = GetImpliedParams(dutchSubsample = False, fixedRRs = True)

paramsDict['Lbar'] = .09

'severe'
paramsDict['k_bar'] = k_bar_min.index[0] #.16% severe 0.075 # np.sum(paramsDict['O-SII rates']*paramsDict['wts']) #  #
myPDmod_sev = PDmodel('min ES', paramsDict, True)
'moderate'
paramsDict['k_bar'] = k_bar_min2.index[0] #14.4% in the mild case; OLD :   0.0875 # np.sum(paramsDict['O-SII rates']*paramsDict['wts']) #  #
myPDmod_mod = PDmodel('min ES', paramsDict, True)
'mild'
paramsDict['k_bar'] = k_bar_min3.index[0] #14.4% in the mild case; OLD :   0.0875 # np.sum(paramsDict['O-SII rates']*paramsDict['wts']) #  #
myPD_mild = PDmodel('min ES', paramsDict, True)


dfTable = pd.DataFrame(index =paramsDict['Names'], columns = ['Country','Names','k','k_micro','k_str_mild','k_str_mod','k_str_sev'])
#dfTable['O-SII Rate'] =  paramsDict['O-SII rates']*100
dfTable['k'] = mySRparams.DataSet.capitalRatio.iloc[0]
dfTable['k_micro'] = 7 + mySRparams.DataSet.banks['p2r  CET1']*100
dfTable['Country'] =  mySRparams.DataSet.banks.Country
dfTable['w'] = mySRparams.wts.iloc[0]*100


dfTable['k_str_mild'] = myPD_mild.dict['k_str'] *100
dfTable['k_str_mod'] = myPDmod_mod.dict['k_str'] *100
dfTable['k_str_sev'] = myPDmod_sev.dict['k_str'] *100
#k_macro_curr = dfTable['k'].values/100 - dfTable['k_micro']/100
dfTable['Names'] = dfTable.index

dfTable.sort_values(['Country','Names'],  ascending = [True, True], inplace = True)
dfTable.to_excel('dfTable_Oct.xlsx')

'''
dfTable['DirectCost_mild'] = myPDLp4.dict['DirectCost']
dfTable['IndirectCost_mod'] = myPDLp4.dict['IndirectCost']
dfTable['IndirectCost_sev'] = myPDLp4.dict['IndirectCost']

dfTable['PD'] = mySRparams.dfPD.iloc[0]
dfTable['PD_Lp4'] = myPDLp4.dict['PD'].T
dfTable['PD_Lp5'] = myPDLp5.dict['PD'].T

'''


#%% plot k vs k_str

hatch_patterns = ['', '//', '-', '.', '']
color_patterns = ['black','white','grey','white']
#color_patterns = ['grey','g','tab:orange']

ax = myplot_frame((12, 6))
dfTable[['k','k_str_mild','k_str_mod','k_str_sev']].plot.bar(align='center',color= color_patterns, edgecolor='black',alpha=.75, ax = ax)
# Apply hatch patterns based on bar color
plt.ylabel(r'$k (\%)$', fontsize = 18)

df = dfTable[['k','k_str_mild','k_str_mod','k_str_sev']]
for i, (col, color) in enumerate(zip(df.columns[1:], color_patterns)):
    for j, bar in enumerate(ax.patches[i * len(df):(i + 1) * len(df)]):
        if color == color_patterns[0]:
            bar.set_hatch(hatch_patterns[0])
        if color == color_patterns[1]:
            bar.set_hatch(hatch_patterns[1])
        if color == color_patterns[2]:
            bar.set_hatch(hatch_patterns[2])

plt.xticks(fontsize=18,rotation=75)  # Adjust font size here
plt.yticks(fontsize=18)  # Adjust font size here
plt.gca().xaxis.set_tick_params(which='minor', size=0)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().tick_params(axis='x', which='minor', bottom='off')
plt.legend([r'Current $k_i$',r'$k_i^*$(Milder)', r'$k_i^*$(Moderate)', r'$k_i^*$(Severe)'], fontsize = 14,loc='upper center', ncol=4)

# Add vertical lines between different categories
categories = dfTable['Country'].unique()

count = 0 
country_count = 'abs'
for idx, df in dfTable.iterrows():
    if df.Country == country_count:
        continue
    country_count = df.Country
    line_position = dfTable.index.get_loc(dfTable.index[dfTable.Country == country_count][-1]) + 0.5
    plt.axvline(line_position, color='gray', linestyle='--')
    # annotate
    plt.annotate(df.Country[:2], (line_position- .5, 30), textcoords="offset points", xytext=(0,10), ha='center', fontsize=25)    

saveFig(path,'optKbar_col') 




'''
dfTable.sort_values(by='k_macro_str',ascending=False).plot.bar(figsize=(12,6), cmap = 'tab20')
plt.legend(['O-SII Rate',r'$k_{macro,i}^*$'])
plt.xlabel('')
saveFig(path,'k_Euro_ES') 
'''

#%% plot SCD of the opt

#dfTable[['PD','PD_Lp5']].plot.bar(figsize = (10,10))
dfTable = pd.DataFrame(index =paramsDict['Names'], columns=['SCD_opt','SCD_init'])
dfTable['w'] = mySRparams.wts.iloc[0]*100

'Run Optimized'
paramsDict['k_bar'] = 0.144 #k_bar_min2.index[0] #14.4% in the mild case; OLD :   0.0875 # np.sum(paramsDict['O-SII rates']*paramsDict['wts']) #  #
myPD_mod = PDmodel('min ES', paramsDict, True)
dfTable['SCD_opt'] = myPD_mod.dict['SCD']*100

'Get Initial'
dfTable['k_micro'] = 7 + mySRparams.DataSet.banks['p2r  CET1']*100
dfTable['k'] = mySRparams.DataSet.capitalRatio.iloc[0]
k_macro_curr = dfTable['k'].values/100 - dfTable['k_micro']/100

dfTable['IndirectCost'], dfTable['DirectCost'], dfTable['SCD_init'] = myPDmod_mod.getResidMulti(k_macro_curr,True)
dfTable['SCD_init'] = dfTable['SCD_init'] *100

paramsDict['k_bar'] = 0 #k_bar_min2.index[0] #14.4% in the mild case; OLD :   0.0875 # np.sum(paramsDict['O-SII rates']*paramsDict['wts']) #  #
myPDmod_micro = PDmodel('min ES', paramsDict, True)

'Plot'
ax = myplot_frame((18, 6))
plt.scatter(dfTable['w'],dfTable['SCD_init'],label='Initial', color='blue')
plt.scatter(dfTable['w'],dfTable['SCD_opt'], label='Optimized', color='red', marker ='x')
#plt.scatter(dfTable['w'],dfTable['SCD_Lp5'], label='Optimized L50', color='green')
# Connect the points with lines
for i in range(len(dfTable)):
    plt.plot([dfTable['w'][i], dfTable['w'][i]], [dfTable['SCD_init'][i], dfTable['SCD_opt'][i]], color='gray')
    plt.annotate('', xy=(dfTable['w'][i], dfTable['SCD_opt'][i]), xytext=(dfTable['w'][i], dfTable['SCD_init'][i]),
                 arrowprops={'arrowstyle': '-|>', 'color': 'gray'})

#for index, row in dfTable[['w','PD']].iterrows():
#    plt.annotate(index, (row['w'], row['PD']), textcoords="offset points", xytext=(0, 10), ha='center')
plt.legend(fontsize = 18)
plt.xlabel(r'$w(\%)$',fontsize = 20)
plt.ylabel(r'$SCD(\%)$',fontsize = 20)
saveFig(path,'SCD_opt') 



#%%
###########################################################
'Table with descriptive statistics : Table 3 Model Input Data'
table = pd.DataFrame(index=paramsDict['Names'], columns = ['Country','Code','Name','w','CDS', 'PD','rho1', 'rho2', 'rho3','sigma_hat', 'k', 'k_p2r']) #, 'k_macro'
j = 0
for index, row in table.iterrows():
    row['Country'] = mySRparams.DataSet.banks.loc[index,'Country']
    row['Code'] = index
    row['Name'] =mySRparams.DataSet.banks.loc[index,'Bank Name']
    row['w'] = np.round(paramsDict['wts'][j]*100,2)
    row['CDS'] = mySRparams.DataSet.CDSprices.loc[mySRparams.evalDate,index].values[0]
    row['PD'] = mySRparams.dfPD.loc[mySRparams.evalDate,index].values[0]*100
    #row['LGD'] = paramsDict['LGD'][j]*100
    row['sigma_hat'] = paramsDict['Sigma'][j]*100
    row['rho1'] = paramsDict['Rho'][j][0]
    row['rho2'] = paramsDict['Rho'][j][1]
    row['rho3'] = paramsDict['Rho'][j][2]
    row['k'] = mySRparams.DataSet.capitalRatio.loc[mySRparams.evalDate,index].values[0]
    row['k_p2r'] = paramsDict['k_p2r'][j]*100
    #row['k_macro'] = myPD.dict['k_macro_str'][j]
    j = j +1 

table.sort_values(by ='Country').to_excel('Table3ModelInputData.xlsx')
print(table.round(2).sort_values(by ='Country').to_latex())    


#%%% Load Data
'''
mySRparams = GetImpliedParams(dutchSubsample = True, fixedRRs = True)
paramsDict = {}
paramsDict['Sigma'], paramsDict['wts'], paramsDict['LGD'], paramsDict['Rho'], paramsDict['Names'] = \
    mySRparams.dfSigmaEval.values.astype(np.float64), mySRparams.wtsEval.values.astype(np.float64), mySRparams.LGDEval.values.astype(np.float64), mySRparams.ldngs.astype(np.float64), mySRparams.dfSigmaEval.index
'''



'Laeven/Vicera DB'

dfCrises = pd.read_csv('crisesLaevenVicera.csv', index_col='Country')
dfCrises.plot.scatter(y='Output loss',x='Recap cost')
