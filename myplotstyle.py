''' Several useful plot style appropriate for academic papers
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import numpy as np


def myplot_frame(figSize=(4, 3),fntsize=30):
    'plot function parameters'
    fig = plt.figure(figsize=figSize)
    ax = fig.add_subplot(1, 1, 1)
    plt.rc('font', family='serif', size=fntsize) #
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    return ax

def myplot(x,y,labels, NumFormat, yLim=False, lineStyle='-'):
    '''
    lineStyle : 'dashed' or 'solid'
    labels : a collection two strings 'string', xlabel and ylabel
    title: a 'string'
    '''
    xlabel, ylabel = labels #, lineLabel
    ax = myplot_frame((8, 6))
    ax.plot(x, y,  color='k', ls=lineStyle) #, label = r'$t^*$' , label=lineLabel 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FormatStrFormatter(NumFormat))

    #ax.legend()
    
    if yLim != False:
        ax.set_ylim(yLim)

    return ax

def myplotNLines(x,yy, ylbl, lbls, xlbl, NumFormat='%.3f',yLim=False):

    #plots up to 4 lines
    clrs = ('k', 'c', 'g','r','c')
    linStyles = ('-','--',':','-.','-.' )
    ax = myplot_frame((8, 6))
    ax.yaxis.set_major_formatter(FormatStrFormatter(NumFormat))
    for j, (y, clr,linStyle,lbl )in enumerate(zip(yy,clrs ,linStyles,lbls )):
        ax.plot(x, y,  color=clr, ls=linStyle, label=lbl) #, label = r'$t^*$'
        # ax.plot(x, yy[1],  color=clrs(1), ls=('-.'), label=lbls[1]) #, label = r'$t^*$'
        # ax.plot(x, yy[:,2],  color='k', ls=(':'), label='p='+str(round(.5,2))) #, label = r'$t^*$'
    ax.set_xlabel(xlbl, fontsize=25)
    ax.set_ylabel(ylbl)
    
#    ax.set_title(ttle)

    if len(yy)>1:
        ax.legend()
    if yLim != False:
        ax.set_ylim(yLim)

    return ax


def myBarChart(A,B,C,xAxis):
    N = len(xAxis)
    ind = np.arange(N)    # the x locations for the groups
    plt.bar(ind, A, width=1, color = 'b')
    plt.bar(ind, B, width=1, color = 'g', bottom = A)
    plt.bar(ind, C, width=1, color = 'r', bottom = A + B)
    plt.xticks(ind, np.around(tau_s_grid,3))
    ax = plt.axes()
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    # Show graphic
    plt.show()
    return

def myplotSP(optimalValues):
    'plot from dictionary'
    xi = optimalValues['xi']

    Hh = optimalValues['H pts']
    plt.plot(xi,Hh)
    plt.title('H fn')
    plt.show()


    d_pl, d_min = optimalValues['d+'], optimalValues['d-']
    plt.plot(xi,d_pl, label='buy')
    plt.plot(xi, d_min, label='sell')
    plt.legend()
    plt.title(r'$\sigma=$'+str(round(sigma,2)))
    plt.show()

    s = optimalValues['s']
    d_pl, d_min = optimalValues['d+'], optimalValues['d-']
    c_y, c_o = optimalValues['c_y'], optimalValues['c_o']
    xi_after = xi + d_pl - d_min
    m = 1- (xi_after + s + c_y + c_o)

    lbls = [r'$\xi$', 's', r'$c_y$',r'$c_o$', 'm']

    #myplot_frame()
    plt.title(r'$\sigma=$'+str(round(sigma,2)))
    plt.xlabel(r'$\xi$')
    plt.stackplot(xi,[xi_after, s, c_y, c_o, m], labels =lbls, alpha=0.4)   #colors = pal
    plt.legend()
    plt.show()

def my_3dplot(x,y,z,Plotlabel,lbls=('W','X'),transpose=False):
    '''
    This expects that V will have the x-axis on rows, and y-axis on the columns unless you transpose'''
    xv, yv = np.meshgrid(x,y)

    if transpose ==True:
        z = z.T

    fig = plt.figure(figsize=(12, 9))
    #ax = fig.add_subplot(1, 1, 1)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xv,yv,z)
    ax.set_xlabel(lbls[0])
    ax.set_ylabel(lbls[1])
    ax.set_title(Plotlabel)

    return ax

def my_3dContour(x,y,z,ttle,lbls, plotValue=False ,yLabel=True):
    xx, yy = np.meshgrid(x,y)

    myplot_frame((8, 6),fntsize=20)
    if plotValue == False:
        contours = plt.contour(xx, yy, z,colors="black")
    else:
        contours = plt.contour(xx, yy, z,plotValue, colors="black")
        
    plt.clabel(contours, inline=True, fontsize=10)
    'here I could add some colors to negative values, etc.'
    #plt.imshow(D_pl_opt, extent=[0, 1, 0, 1], origin='lower',
    #           cmap='RdGy', alpha=0.25)
    #plt.colorbar();
    plt.xlabel(lbls[0])
    if yLabel==True:
        plt.ylabel(lbls[1])
    plt.title(ttle)
    
    return ax

def my_3dContour2plots(x,y,z1,z2,ttle, lbls, plotValue=False ):

    ax = myplot_frame((8, 6),fntsize=20)
    
    contours = plt.contour(x,y,z1,[np.round(plotValue,3)], colors='black')
    plt.clabel(contours, inline=True, fontsize=15)
    contours = plt.contour(x,y,z2,[np.round(plotValue,3)], colors='grey')
    plt.clabel(contours, inline=True, fontsize=15)
    
    plt.xlabel(lbls[0])
    plt.ylabel(lbls[1])
    plt.title(ttle)
    
    plt.annotate(r'$SCD_1=SCD_{ref}$',(.2,.1))
    plt.annotate(r'$SCD_2=SCD_{ref}$',(.08,.2))    
    plt.show()        
    

def myplotThreeLines(x,yy, ttle):
    '''
    lineStyle : 'dashed' or 'solid'
    labels : a collection two strings 'string'
    ttle: title, a 'string'
    yy : Nx3 arrat
    '''
    #xlabel, ylabel, lineLabel = labels
    ax = myplot_frame()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.plot(x, yy[:,0],  color='k', ls=('solid'), label='p='+str(round(1,2))) #, label = r'$t^*$'
    ax.plot(x, yy[:,1],  color='k', ls=('dashed'), label='p='+str(round(.8,2))) #, label = r'$t^*$'
    ax.plot(x, yy[:,2],  color='k', ls=(':'), label='p='+str(round(.5,2))) #, label = r'$t^*$'


    ax.set_xlabel(r'$\tau_x$')
    ax.set_title(ttle)
    ax.legend()

    return ax

def myplotNLines_v02(x,yy, ttle, lbls,xlbl,clrs, linStyles, NumFormat='%.3f',yLim=False):
    '''
    here you can also provide the line styles and colors
    ''' 
    #plots up to 4 lines
    ax = myplot_frame()
    ax.yaxis.set_major_formatter(FormatStrFormatter(NumFormat))
    for j, (y, clr,linStyle,lbl )in enumerate(zip(yy,clrs ,linStyles,lbls )):
        ax.plot(x, y,  color=clr, ls=linStyle, label=lbl) #, label = r'$t^*$'
        # ax.plot(x, yy[1],  color=clrs(1), ls=('-.'), label=lbls[1]) #, label = r'$t^*$'
        # ax.plot(x, yy[:,2],  color='k', ls=(':'), label='p='+str(round(.5,2))) #, label = r'$t^*$'


    ax.set_xlabel(xlbl)
    ax.set_title(ttle)

    if len(yy)>1:
        ax.legend()
    if yLim != False:
        ax.set_ylim([0,0.12])

    return ax

def lollipopChart(df,myColor='grey'):
    #fig, ax = plt.subplots(figsize=(6,6), dpi= 80)
    '''
    ax : either pre-specified plot axis or False otherwise
    color : aternatively use firebrick
    '''
    figSize = (15, 3)
    
    fig = plt.figure(figsize=figSize)
    ax = fig.add_subplot(1, 1, 1)
    
    y = df.columns[0]
    #df.sort_values(by=y, ascending=False, inplace=True)
    ax.vlines(x=df.index, ymin=0, ymax=df[y], color=myColor, alpha=0.4, linewidth=2)
    ax.scatter(x=df.index, y=df[y], s=75, color='navy', alpha=0.7)
    
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.index.str.upper(), rotation=90)
    ax.set_ylim(0)
    
    # Annotate
    for row in df.itertuples():
        ax.text(row.Index, row[1], s=round(row[1], 2), horizontalalignment= 'center', verticalalignment='bottom', fontsize=14)
    
    return ax

def lollipopChartVert(df,myColor='grey'):
    #fig, ax = plt.subplots(figsize=(6,6), dpi= 80)
    '''
    ax : either pre-specified plot axis or False otherwise
    color : aternatively use firebrick
    '''
    figSize = (3,9)
    
    fig = plt.figure(figsize=figSize)
    ax = fig.add_subplot(1, 1, 1)
    
    y = df.columns[0]
    #df.sort_values(by=y, ascending=False, inplace=True)
    ax.hlines(y=df.index, xmin=0, xmax=df[y], color=myColor, alpha=0.4, linewidth=2)
    ax.scatter(y=df.index, x=df[y], s=75, color='navy', alpha=0.7)

    ax.set_yticks(df.index)
    ax.set_yticklabels(df.index.str.upper()) #rotation=90
    ax.set_xlim(0)

    # Annotate
    for row in df.itertuples():
        ax.text(row[1], row.Index, s=round(row[1], 2), horizontalalignment= 'center', verticalalignment='bottom', fontsize=12)

    return ax

def lollipopChart2(df):
    'lollipop plot of 2 series from a dataFrame'
    ax = myplot_frame((7,6))
    y1 = df.columns[0]
    y2 = df.columns[1]
    df.sort_values(by=y1, ascending=False, inplace=True)
      
    ax.vlines(x=df.index, ymin=df[y2], ymax=df[y1], color='grey', alpha=0.4)
    ax.scatter(df.index, df[y1], color='navy', alpha=1, label=y1)
    ax.scatter(df.index, df[y2], color='gold', alpha=0.8 , label=y2)
    #ax.set_xticks(df.index)
    ax.set_xticklabels(df.index.str.upper())
    ax.legend()
    
    for row in df[y1].to_frame().itertuples():
        ax.text(row.Index, row[1], s=round(row[1], 2), horizontalalignment= 'center', verticalalignment='bottom')

    for row in df[y2].to_frame().itertuples():
        ax.text(row.Index, row[1], s=round(row[1], 2), horizontalalignment= 'center', verticalalignment='bottom')

    return ax


def saveFig(path,fileName, fileType = '.pdf'):
    plt.savefig(path+fileName+'.png',bbox_inches = 'tight')
    plt.savefig(path+fileName+fileType,bbox_inches = 'tight')

def mySpiderCharts(): 
    'fix later'
    labels=universeFull
    stats=100*covarr.ES[universeFull].loc[[.99]].T.values

    angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))

    fig=plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    #ax.set_title('Exposure Shortfall')
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=10)
    fig.legend([r'$q=.99$'], loc = 'lower center', frameon=False, ncol=3)
    
    
def myScatter(dfx, dfy, myLabels, setLegend=False, setLabels=True):
    '''
    dfx, dfy : series for the scatterplot
    myLabels : tuple for x label and y label 
    '''
    ax = myplot_frame((15,8))
    ax.scatter(dfx,dfy, label = myLabels[2], color =myLabels[3] )
    ax.set_xlabel(myLabels[0])
    ax.set_ylabel(myLabels[1])
    ax.set_ylim(0)
    ax.set_xlim(0)
    
    if setLabels==True:
        for Name, row in dfx.iteritems():
            ax.text(row,dfy[Name], Name, fontsize=15)
        
    if setLegend == True:
        ax.legend()
        
    return ax

def myHeatmapDoubleM(df):
    'Heatmap matrix, carving out the diagonal'
    ax= myplot_frame((7,6)) #plt.subplot(111, polar=True)    
    mask = np.invert(np.ones_like(df, dtype=np.bool))    
    np.fill_diagonal(mask,True)
    sns.heatmap(df.round(2), mask=mask, annot=True, fmt=".3g", cmap='Oranges', vmin=0, vmax=df[np.invert(mask)].max().max(), ax=ax)
    return ax
