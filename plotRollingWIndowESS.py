# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:06:59 2024

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from setParams import SetParams
from myplotstyle import * 



def plotRWess(df1, df2,myLegend,myLabels):
    
    axs = myplot_frame((20,8))
    #rwcovar.ES99['Sys'].plot(color='black',ax=axs,label='ESS, 99%')
    df1.plot(color='black',linestyle='-', ax=axs,label=labels[0])
    axs.set_ylabel("ESS/VSTOX", fontsize='small')
    df2.plot(color='grey',linestyle='--',ax=axs, label=labels[1])
    axs.set_xlabel('')
    #axs.set_ylabel("VSTOXX", fontsize='small')
    #axs.legend([], fontsize='xx-small', loc ='upper right')
    axs.legend(myLegend, fontsize='xx-small')

    plt.axvline(x='2007-08-09', color='grey') #BNP anounces CDO losses  
    plt.text('2007-08-09',60,'(a)',rotation=90, fontsize='small')

    plt.axvline(x='2007-09-14', color='grey') #Norhtern rock  
    plt.text('2007-09-14',10,'(b)',rotation=90, fontsize='small')

    plt.axvline(x='2008-03-11', color='grey') #Bear Stearns  
    plt.text('2008-03-11',10,'(c)',rotation=90, fontsize='small')
    plt.axvline(x='2008-09-15', color='grey') #Lehman   
    plt.text('2008-09-15',10,'(d)',rotation=90, fontsize='small')

    plt.axvline(x='2012-05-24', color='grey') #draghi courageous leap  
    plt.text('2012-05-24',10,'(e)',rotation=90, fontsize='small')

    plt.axvline(x='2012-07-26', color='grey') #draghi whatever it takes 
    plt.text('2012-07-26',20,'(f)',rotation=90, fontsize='small')

    plt.axvline(x='2015-06-30', color='grey') #Greece misses IMF payment
    plt.text('2015-06-30',10,'(g)',rotation=90, fontsize='small')

    plt.axvline(x='2016-06-23', color='grey') #Brexit referendum  
    plt.text('2016-06-23',10,'(h)',rotation=90, fontsize='small')

    plt.axvline(x='2020-03-09', color='grey') #first covid lockdown, Europe (Italy)  
    plt.text('2020-03-09',10,'(i)',rotation=90, fontsize='small') 
    plt.axvline(x='2022-02-20', color='grey') #russian invasion in Ukraine 
    plt.text('2022-02-20',10,'(j)',rotation=90, fontsize='small') #russian invasion in Ukraine
    #ax2 = axs.twinx()
    #df2.plot(color='blue',linestyle='--',ax=ax2, label='CISS Index')
    #ax2.legend(['CISS'], fontsize='xx-small',loc='upper center')
    return axs

def plotRWess2ax(df1, df2,myLegend,myLabels):
    #df1, df2, myLegend, myLabels= ESScds, ESSeq,['CDS-based','Equity-based'],['ESS(CDS)','ESS(EQ)']
    
    axs = myplot_frame((15,5))
    #rwcovar.ES99['Sys'].plot(color='black',ax=axs,label='ESS, 99%')
    df1.plot(color='black',linestyle='-', ax=axs,label=myLegend[0])
    axs.set_ylabel(myLabels[0], fontsize='small')#myLabels[0]+'/'+myLabels[1]
    #df2.plot(color='grey',linestyle='--',ax=axs, label=labels[1])
    axs.set_xlabel('')
    #axs.set_ylabel(, fontsize='small')
    #axs.legend([], fontsize='xx-small', loc ='upper right')
    axs.legend(fontsize='xx-small')

    plt.axvline(x='2007-08-09', color='grey') #BNP anounces CDO losses  
    plt.text('2007-08-09',60,'(a)',rotation=90, fontsize='small')

    plt.axvline(x='2007-09-14', color='grey') #Norhtern rock  
    plt.text('2007-09-14',50,'(b)',rotation=90, fontsize='small')

    plt.axvline(x='2008-03-11', color='grey') #Bear Stearns  
    plt.text('2008-03-11',10,'(c)',rotation=90, fontsize='small')
    plt.axvline(x='2008-09-15', color='grey') #Lehman   
    plt.text('2008-09-15',10,'(d)',rotation=90, fontsize='small')

    plt.axvline(x='2012-05-24', color='grey') #draghi courageous leap  
    plt.text('2012-05-24',10,'(e)',rotation=90, fontsize='small')

    plt.axvline(x='2012-07-26', color='grey') #draghi whatever it takes 
    plt.text('2012-07-26',20,'(f)',rotation=90, fontsize='small')

    plt.axvline(x='2015-06-30', color='grey') #Greece misses IMF payment
    plt.text('2015-06-30',10,'(g)',rotation=90, fontsize='small')

    plt.axvline(x='2016-06-23', color='grey') #Brexit referendum  
    plt.text('2016-06-23',10,'(h)',rotation=90, fontsize='small')

    plt.axvline(x='2020-03-09', color='grey') #first covid lockdown, Europe (Italy)  
    plt.text('2020-03-09',10,'(i)',rotation=90, fontsize='small') 
    plt.axvline(x='2022-02-20', color='grey') #russian invasion in Ukraine 
    plt.text('2022-02-20',10,'(j)',rotation=90, fontsize='small') #russian invasion in Ukraine
    ax2 = axs.twinx()
    df2.plot(color='blue',linestyle='--',ax=ax2, label=myLegend[1])
    ax2.set_ylabel(myLabels[1], fontsize='small')#myLabels[0]+'/'+myLabels[1]    
    ax2.legend(fontsize='xx-small',loc='upper center')

    return axs

