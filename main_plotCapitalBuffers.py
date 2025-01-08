# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 22:42:12 2023

@author: danie
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from myplotstyle import * 

path = "C:\\Users\\danie\\Dropbox\\Amsterdam\\Systemic Risk Europe DNB\\imagesTemp\\"


'load values'
dfTable = pd.read_csv('OsiiEeiESS.csv', index_col='Short Code')
dfTable.drop('EEI', axis=1, inplace = True)

#dfTable = pd.read_csv('OsiiESS.csv', index_col='Short Code')


hatch_patterns = ['', '//', '.', '-', '']
#color_patterns = ['grey','g','tab:orange']
color_patterns = ['black','white','grey']

#hatch_patterns = ['.', '//.', '', '-', '']
#color_patterns = ['grey','white','black']


'plot'
ax = myplot_frame((12, 6))
dfTable.plot.bar(align='center',color= color_patterns, edgecolor='black',alpha=.75, ax = ax)
plt.ylabel(r'$k_{i,macro}(\%)$', fontsize = 21)
plt.xlabel('')


'Apply hatch patterns based on bar color'
for i, (col, color) in enumerate(zip(dfTable.columns[1:], color_patterns)):
    for j, bar in enumerate(ax.patches[i * len(dfTable):(i + 1) * len(dfTable)]):
        if color == color_patterns[0]:
            bar.set_hatch(hatch_patterns[0])
        elif color == color_patterns[1]:
            bar.set_hatch(hatch_patterns[1])
        elif color == color_patterns[2]:
            bar.set_hatch(hatch_patterns[2])            

plt.xticks(fontsize=18,rotation=75)  # Adjust font size here
plt.yticks(fontsize=21)  # Adjust font size here
plt.gca().xaxis.set_tick_params(which='minor', size=0)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().tick_params(axis='x', which='minor', bottom='off')
plt.legend(['O-SII Rate','ESS Approach'], fontsize = 14,loc='lower center', ncol=3,bbox_to_anchor=(0.5, -0.32))
#plt.legend(['ESS (Euro-wide)','ESS (Country-wide)'], fontsize = 14,loc='upper center', ncol=3,bbox_to_anchor=(0.5, -0.25))

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
    plt.annotate(df1.Country[:2], (line_position- .5, 2.5), textcoords="offset points", xytext=(0,2.9), ha='center', fontsize=18)    

saveFig(path,'OsiiModelRank_loc') 
#saveFig(path,'OsiiModelRank_eur') 