import pandas as pd
import numpy as np
from copy import deepcopy
from bokeh.palettes import GnBu, RdPu, viridis
from math import pi
#import geopandas as gpd
import json

from bokeh.models import (ColumnDataSource, Select, HoverTool, FactorRange, Panel,Tabs, LabelSet, Label, StringFormatter,
                          NumeralTickFormatter, DatetimeTickFormatter,LinearInterpolator, Legend,GeoJSONDataSource,
                         ColorBar, LinearColorMapper, BasicTicker, PrintfTickFormatter)

from bokeh.models import LinearAxis, CategoricalAxis


from bokeh.transform import dodge, jitter, factor_cmap
from bokeh.plotting import figure, save, reset_output
import matplotlib.pyplot as plt
from bokeh.io import output_notebook, show, output_file, export_png
from bokeh.io import export_svgs
from shapely.geometry import Point
from bokeh.layouts import column, row, gridplot 

import ipywidgets as widgets
from ipywidgets import interact, interactive



import seaborn as sns
sns.set_style('white')


plt.rcParams["figure.figsize"] = [16,9]

np.warnings.filterwarnings('ignore')

    
def dfTeamsChamp(year):
    #Function from 2004 onwards
    
    dfR = pd.read_csv('data/races.csv', index_col=0, delimiter=',')
    dfR = dfR[dfR['year'] == int(year)]
    dfR.drop('url', axis=1, inplace=True)
    dfR.sort_values('date', inplace=True)
    
    
    #df contructors teams
    dfC = pd.read_csv('data/constructors.csv', delimiter=',', index_col=0)
    
    #df constructor standings
    dfC_S = pd.read_csv('data/constructor_results.csv', delimiter=',', index_col=0)
    dfC_S.set_index('raceId', inplace=True)
    #dfC_S = pd.read_csv( 'constructor_standings.csv', index_col=1, delimiter=',')
    #filter year qualify
    CSF = np.array(dfR.index)
    dfC_S1 = dfC_S.index.isin(CSF)
    dfC_S1 = dfC_S[dfC_S1]
    
    c_F = np.array(dfC_S1.constructorId.unique())
    dfC1 = dfC.index.isin(c_F)
    dfC1 = dfC[dfC1]
    
    #dictionary constructorId x constructor name
    diC_N = pd.Series(dfC1.name.values, dfC1.index.values).to_dict()
    
    #add constructor name to df constructor standings
    dfC_S1['Team'] = dfC_S1.constructorId.map(diC_N)
    
    dfRes = pd.read_csv('data/results.csv', index_col='raceId', sep=',')
    dfResf = dfRes.index.isin(CSF)
    dfRes = dfRes[dfResf]
    dfRes['Team'] = dfRes.constructorId.map(diC_N)
    dfRes.replace('\\N', 0, inplace=True)
    dfRes['rank'] =  dfRes['rank'].astype(int)
    
    
    #df qualis
    dfQ = pd.read_csv('data/qualifying.csv', index_col=1, delimiter=',')
    #filter year qualify
    qF = np.array(dfR.index)
    dfQ1 = dfQ.index.isin(qF)
    dfQ1 = dfQ[dfQ1]
    dfQ1['Team'] = dfQ1.constructorId.map(diC_N)
    
    #df drivers
    dfD = pd.read_csv('data/drivers.csv', index_col=0, delimiter=',')
    dfD.drop('url', axis=1, inplace=True)
    #filter drivers year
    dfDY = dfQ1.set_index('driverId')
    dfDYF = np.array(dfDY.index.unique())
    dfDY = dfD.index.isin(dfDYF)
    dfDY = dfD[dfDY]
    #dict driverId x driverName
    diDidxDN = pd.Series(dfDY.driverRef.values, dfDY.index.values).to_dict()
       
    #add driver name to df quali
    dfQ1['DN'] = dfQ1.driverId.map(diDidxDN)
    
    #dict DriverName x Team
    diDxT = pd.Series(dfQ1.Team.values, dfQ1.DN.values).to_dict()
    
    
    #dict DriverName x TeamID
    diCxD = pd.Series(dfQ1.constructorId.values, dfQ1.DN.values).to_dict()
    
    #df driver standings champioship
    dfD_C = pd.read_csv('data/driver_standings.csv', sep=',', index_col='raceId')
    dfD_CF = dfD_C.index.isin(qF)
    dfD_C = dfD_C[dfD_CF]
    dfD_C['DN'] = dfD_C.driverId.map(diDidxDN)
    dfD_C['Team'] = dfD_C.DN.map(diDxT)
    
       
    #costructor cham
    dfC_S2 = dfC_S1.sort_values('points', na_position ='first')
    dfC_S2['cumsum'] = pd.DataFrame(dfC_S2.groupby('Team')['points'].cumsum())
    dfC_S3 = pd.DataFrame(dfC_S2.groupby('Team')['points'].sum())
    dfC_S3.sort_values('points', ascending=False, inplace=True)
    dfC_S3['c'] = dfC_S3.index.map({'Red Bull': '#121F45', 'Ferrari':'#A6051A', 'Mercedes':'#00A19C',
                                       'Alpine F1 Team': '#005BA9', 'Haas F1 Team': '#F9F2F2', 'Williams': '#005AFF',
                                       'AlphaTauri': '#00293F','McLaren': '#FF8000', 'Alfa Romeo': '#981E32','Aston Martin':'#00352F',
                                   'Racing Point':'#F596C8', 'Renault':'#FFF500', 'Toro Rosso':'#469BFF', 'Catherham':'#048646',
                                   'Sauber':'#9B0000','Force India':'#F596C8','Lotus F1':'#FFB800', 'Marussia':'#6E0000',
                                   'Jaguar':'#08623e', 'Toyota':'#8e0018', 'BAR':'#d5d5d3', 'Jordan':'#ead136',
                                   'Prost':'#03033f', 'Super Aguri': '#b61211', 'Benetton':'#72dffe', 'Arrows':'#fc8d2c',
                                   'Honda':'#cccfc8', 'BMW Sauber':'#ffffff', 'Brawn':'#cccfc8'})
    
    pos = ['1st', '2nd', '3rd', '4th', '5th','6th','7th','8th','9th','10th','11th','12th','13th','14th']   
    dfC_S3['pos'] = pos[:len(dfC_S3.index)]    
    wins = dfD_C[dfD_C.wins >= 1].groupby(['raceId', 'Team'])['wins'].sum().reset_index()
    dfC_S3['wins'] = wins[wins.raceId == wins.raceId.max()].set_index('Team').drop('raceId', axis=1)
    dfC_S3['wins'].fillna(0, inplace=True)
    dfC_S3['poles'] = dfQ1[dfQ1.position == 1].groupby('Team')['position'].count()
    dfC_S3.poles.fillna(0, inplace=True)
    dfC_S3['FastLap'] = dfRes[dfRes['rank'] == 1].groupby('Team')['rank'].sum()
    dfC_S3['FastLap'] = dfC_S3['FastLap'].fillna(0)
    
    return dfC_S3

def dfLineChart(year):
    #Function from 2003 onwards
    
    dfR = pd.read_csv('data/races.csv', index_col=0, delimiter=',')
    dfR = dfR[dfR['year'] == int(year)]
    dfR.drop('url', axis=1, inplace=True)
    dfR.sort_values('date', inplace=True)
    dfR.name = dfR.name.str.replace('Grand Prix', '')
    
    
    #df contructors teams
    dfC = pd.read_csv('data/constructors.csv', delimiter=',', index_col=0)
    
    #df constructor standings
    dfC_S = pd.read_csv('data/constructor_results.csv', delimiter=',', index_col=0)
    dfC_S.set_index('raceId', inplace=True)
    #dfC_S = pd.read_csv( 'constructor_standings.csv', index_col=1, delimiter=',')
    #filter year qualify
    CSF = np.array(dfR.index)
    dfC_S1 = dfC_S.index.isin(CSF)
    dfC_S1 = dfC_S[dfC_S1]
    
    c_F = np.array(dfC_S1.constructorId.unique())
    dfC1 = dfC.index.isin(c_F)
    dfC1 = dfC[dfC1]
    
    #dictionary constructorId x constructor name
    diC_N = pd.Series(dfC1.name.values, dfC1.index.values).to_dict()
    #dict raceId x GP name
    dicRxG = pd.Series(dfR.name.values, dfR.index.values)
    #dict colors x teams
    diC = {'Red Bull': '#121F45', 'Ferrari':'#A6051A', 'Mercedes':'#00A19C','Alpine F1 Team': '#005BA9', 'Haas F1 Team': '#F9F2F2', 'Williams': '#005AFF',
            'AlphaTauri': '#00293F','McLaren': '#FF8000', 'Alfa Romeo': '#981E32','Aston Martin':'#00352F','Racing Point':'#F596C8','Renault':'#FFF500',
           'Toro Rosso':'#469BFF', 'Catherham':'#048646', 'Sauber':'#9B0000','Force India':'#F596C8','Lotus F1':'#FFB800', 'Marussia':'#6E0000',
            'Jaguar':'#08623e', 'Toyota':'#8e0018', 'BAR':'#d5d5d3', 'Jordan':'#ead136','Prost':'#03033f', 'Super Aguri': '#b61211', 'Benetton':'#72dffe', 'Arrows':'#fc8d2c',
            'Honda':'#cccfc8', 'BMW Sauber':'#ffffff', 'Brawn':'#cccfc8', 'Lotus':'#FFB800'}
    
    #add constructor name to df constructor standings
    dfC_S1['Team'] = dfC_S1.constructorId.map(diC_N)
    dfC_S1['GP'] = dfC_S1.index.map(dicRxG)
    dfC_S1 = pd.DataFrame(dfC_S1.groupby(['Team','GP'], sort=False)['points'].sum())
    dfC_S1['c'] = dfC_S1.index.get_level_values('Team').map(diC).fillna('white')
    d = []
    gps = dfC_S1.sort_values('points', ascending=True).index.get_level_values('GP').unique()
    cl = pd.Series(dfC_S1.index.get_level_values('Team').unique().values, range(0, len(dfC_S1.index.get_level_values('Team').unique()))).to_dict()

    for p in dfC_S1.groupby('Team', sort=False)['points']:
        np.array(d.append(p[1].values)).flatten()
        df = pd.DataFrame(d, columns=gps)
        df = df.T
        df.fillna(0, inplace=True)
        df.rename(columns = cl, inplace = True)
        df = df.cumsum()
        
    p1 = figure(x_range=np.array(df.index), title='Constructor Championship by Grand Prix '+ str(year), plot_height=400, plot_width=750, y_axis_label = 'points',
                 tools='pan, wheel_zoom, box_zoom, reset', toolbar_location='right')
    p1.grid.grid_line_alpha = .15
    p1.grid.grid_line_dash = 'dotted'
    p1.grid.grid_line_dash_offset = 5
    p1.grid.grid_line_width = 1.5
    p1.grid.grid_line_color = 'white'
    p1.axis.major_label_text_font_style = 'bold'
    p1.xaxis.major_label_text_font_size = '5px'
    p1.yaxis.major_label_text_font_size = '11px'
    p1.title.text_font_size = '13.5px'
    p1.title.text_color = 'white'
    p1.outline_line_color=None
    p1.yaxis.axis_label_text_color= 'white'
    p1.axis.major_label_text_color= 'white'
    #p1.xaxis.major_label_orientation = 45
   
    #p1.yaxis.ticker.desired_num_ticks = 12
    p1.toolbar.autohide = True
    p1.y_range.start = -10
    p1.y_range.end = df.values.max() * 1.1

    p1.background_fill_color = '#010727'
    p1.background_fill_alpha =.8
    p1.border_fill_color =  '#010727'
    p1.border_fill_alpha = .8
    p1.axis.axis_line_color ='white'
    p1.axis.minor_tick_line_color ='white'
    p1.axis.major_tick_line_color ='white'
    p1.x_range.range_padding = -.02
    
    gps1 = pd.DataFrame(df.iloc[-1]).reset_index()
    gps1.columns = ['Teams','points']
    gps1 = gps1.sort_values('points', ascending=False)
    gps1['c'] = gps1.Teams.map(diC).fillna('white')
    color = np.array(gps1.c)
    gps1 = np.array(gps1['Teams'])
    xh = np.array(df.index)

    
    for col,c in zip(gps1, color):
        p1.line(df.index, df[col], color=c, line_width=3, line_join = 'bevel', legend=str(col))
        p1.legend.border_line_color = None
        p1.legend.label_text_color='white'
        p1.legend.background_fill_color= None
        p1.legend.border_line_alpha=0
        p1.legend.orientation = 'horizontal'
        p1.legend.location = 'top_left'
        p1.legend.background_fill_alpha = 0
        p1.legend.title = '↓Enable/Disable'
        p1.legend.title_text_color = 'white'
        p1.legend.title_text_font_size = '6.5px'
        p1.legend.label_text_font_size = '6.8px'
        p1.legend.click_policy='hide'
        #p1.add_tools(HoverTool( tooltips=[('GP', '@x{custom}'), ('Team', '@y{custom}')], formatters=dict(x=x_custom, y=y_custom)))

    return p1
    
def Constplot(year):
    
    df = dfTeamsChamp(year)
    srcC = ColumnDataSource(df)
    xC = np.array(df.index.unique())

    pCons = figure(x_range=xC, title='Constructor Championship ' + str(year), plot_height=400, plot_width=750, y_axis_label = 'points',
                     tools='pan, wheel_zoom, box_zoom, reset', toolbar_location='right')
    pCons.vbar(top='points', width=.6, x='Team',  source=srcC, line_color='white', line_width = .2, fill_color='c')

    pCons.grid.grid_line_alpha = .25
    pCons.grid.grid_line_dash = 'dotted'
    pCons.grid.grid_line_dash_offset = 5
    pCons.grid.grid_line_width = 2
    pCons.grid.grid_line_color = 'white'
    pCons.axis.major_label_text_font_style = 'bold'
    pCons.xaxis.major_label_text_font_size = '9px'
    pCons.yaxis.major_label_text_font_size = '11px'
    pCons.title.text_font_size = '15px'
    pCons.title.text_color = 'white'
    pCons.outline_line_color=None
    pCons.yaxis.axis_label_text_color= 'white'
    pCons.axis.major_label_text_color= 'white'

    
    pCons.legend.border_line_color = None
    pCons.legend.label_text_color='white'
    pCons.legend.background_fill_color= None
    pCons.yaxis.ticker.desired_num_ticks = 12
    pCons.toolbar.autohide = True
    pCons.y_range.start = 0

    pCons.background_fill_color = '#010727'
    pCons.background_fill_alpha =.8
    pCons.border_fill_color =  '#010727'
    pCons.border_fill_alpha = .8
    pCons.axis.axis_line_color ='white'
    pCons.axis.minor_tick_line_color ='white'
    pCons.axis.major_tick_line_color ='white'
    hv = HoverTool()
    hv.tooltips=[('Team', '@Team'), ('Points', '@points{0.0}'), ('Position', '@pos'),
                 ('Wins','@wins'), ('Poles', '@poles{0}'), ('Fast Laps', '@FastLap')]
    pCons.add_tools(hv)
    
    lGPst = Label(x=0, y=0, x_offset=601.5, y_offset=300, text=str(int(df.wins.sum())),
           text_color='white', text_font_size='25px', text_font_style='bold')
    pCons.add_layout(lGPst)
    
    lGPs = Label(x=0, y=0, x_offset=600, y_offset=280, text='GPs',
               text_color='white', text_font_size='15px', text_font_style='bold')
    pCons.add_layout(lGPs)
    
    
    #plot wins hbar
    dfW = df[df.wins >= 1].sort_values(['wins','pos'], ascending=True)
    srcW = ColumnDataSource(dfW)

    xW = np.array(dfW.index.unique())
    
    pW = figure(y_range=xW, title='Wins ' + str(year), plot_height=150, plot_width=250, x_axis_label = 'wins',
                     tools='pan, wheel_zoom, box_zoom, reset', toolbar_location='right')
    pW.hbar(y='Team', height=.6, right='wins', source=srcW, line_color='white', fill_color='c', line_width=.2)

    pW.grid.grid_line_alpha = .25
    pW.grid.grid_line_dash = 'dotted'
    pW.grid.grid_line_dash_offset = 5
    pW.grid.grid_line_width = 2
    pW.grid.grid_line_color = 'white'
    pW.axis.major_label_text_font_style = 'bold'
    pW.xaxis.major_label_text_font_size = '10px'
    pW.yaxis.major_label_text_font_size = '7.5px'
    pW.title.text_font_size = '12px'
    pW.title.text_color = 'white'
    pW.outline_line_color=None
    pW.xaxis.axis_label_text_color= 'white'
    pW.axis.major_label_text_color= 'white'
    pW.legend.border_line_color = None
    pW.legend.label_text_color='white'
    pW.legend.background_fill_color= None
    pW.toolbar.autohide = True
    pW.x_range.start = 0
    pW.background_fill_color = '#010727'
    pW.background_fill_alpha =.8
    pW.border_fill_color =  '#010727'
    pW.border_fill_alpha = .8
    pW.axis.axis_line_color ='white'
    pW.axis.minor_tick_line_color ='white'
    pW.axis.major_tick_line_color ='white'
    hvW = HoverTool()
    hvW.tooltips=[('Team', '@Team'), ('Wins','@wins'), ('Championship pos', '@pos')]
    pW.add_tools(hvW)
    
    #plot poles hbar
    dfP = df[df.poles >= 1].sort_values(['poles','pos'], ascending=True)
    srcP = ColumnDataSource(dfP)

    xP = np.array(dfP.index.unique())
    
    pP = figure(y_range=xP, title='Poles ' + str(year), plot_height=150, plot_width=250, x_axis_label = 'poles',
                     tools='pan, wheel_zoom, box_zoom, reset', toolbar_location='right')
    pP.hbar(y='Team', height=.6, right='poles', source=srcP, line_color='white', fill_color='c', line_width=.2)

    pP.grid.grid_line_alpha = .25
    pP.grid.grid_line_dash = 'dotted'
    pP.grid.grid_line_dash_offset = 5
    pP.grid.grid_line_width = 2
    pP.grid.grid_line_color = 'white'
    pP.axis.major_label_text_font_style = 'bold'
    pP.xaxis.major_label_text_font_size = '10px'
    pP.yaxis.major_label_text_font_size = '7.5px'
    pP.title.text_font_size = '12px'
    pP.title.text_color = 'white'
    pP.outline_line_color=None
    pP.xaxis.axis_label_text_color= 'white'
    pP.axis.major_label_text_color= 'white'
    pP.legend.border_line_color = None
    pP.legend.label_text_color='white'
    pP.legend.background_fill_color= None
    #pW.yaxis.ticker.desired_num_ticks = 12
    pP.toolbar.autohide = True
    pP.x_range.start = 0
    pP.background_fill_color = '#010727'
    pP.background_fill_alpha =.8
    pP.border_fill_color =  '#010727'
    pP.border_fill_alpha = .8
    pP.axis.axis_line_color ='white'
    pP.axis.minor_tick_line_color ='white'
    pP.axis.major_tick_line_color ='white'
    hvP = HoverTool()
    hvP.tooltips=[('Team', '@Team'), ('Poles', '@poles{0}'), ('Championship pos', '@pos') ]
    pP.add_tools(hvP)
    
    #plot fast laps hbar
    dfF = df[df.FastLap >= 1].sort_values(['FastLap','pos'], ascending=True)
    srcF = ColumnDataSource(dfF)

    xF = np.array(dfF.index.unique())
    
    pF = figure(y_range=xF, title='Fast laps ' + str(year), plot_height=150, plot_width=250, x_axis_label = 'Fast laps',
                     tools='pan, wheel_zoom, box_zoom, reset', toolbar_location='right')
    pF.hbar(y='Team', height=.6, right='FastLap', source=srcF, line_color='white', fill_color='c', line_width=.2)

    pF.grid.grid_line_alpha = .25
    pF.grid.grid_line_dash = 'dotted'
    pF.grid.grid_line_dash_offset = 5
    pF.grid.grid_line_width = 2
    pF.grid.grid_line_color = 'white'
    pF.axis.major_label_text_font_style = 'bold'
    pF.xaxis.major_label_text_font_size = '10px'
    pF.yaxis.major_label_text_font_size = '7.5px'
    pF.title.text_font_size = '12px'
    pF.title.text_color = 'white'
    pF.outline_line_color=None
    pF.xaxis.axis_label_text_color= 'white'
    pF.axis.major_label_text_color= 'white'
    pF.legend.border_line_color = None
    pF.legend.label_text_color='white'
    pF.legend.background_fill_color= None
    #pW.yaxis.ticker.desired_num_ticks = 12
    pF.toolbar.autohide = True
    pF.x_range.start = 0
    pF.background_fill_color = '#010727'
    pF.background_fill_alpha =.8
    pF.border_fill_color =  '#010727'
    pF.border_fill_alpha = .8
    pF.axis.axis_line_color ='white'
    pF.axis.minor_tick_line_color ='white'
    pF.axis.major_tick_line_color ='white'
    hvF = HoverTool()
    hvF.tooltips=[('Team', '@Team'), ('Fast Laps', '@FastLap{0}'), ('Championship pos', '@pos') ]
    pF.add_tools(hvF)
    
    pL = dfLineChart(year)
    
    t1 = Panel(child=pCons, title='Overall')
    t2 = Panel(child =pL, title='By Grand Prix')
    tabs = Tabs(tabs= [t1, t2])
    
    col = column([tabs, row([pW, pP, pF])], spacing=5)
    
    t = Panel(child=col, title=str(year))
    
    return t


def dfWinsPolesY(year):
    df = dfTeamsChamp(year)
    df = df[(df.wins >= 1) | (df.poles >= 1)]
    df[['wins','poles']] = df.iloc[::,-2:].astype(int)
    df.drop('points', axis=1, inplace=True)
    df['year'] = year
    df.reset_index(inplace=True)
    df.set_index('year',inplace=True)
    return df

def dfW_P_All():
    df = dfWinsPolesY(2004)
    years = range(2005,2022)
    for y in years:
        df = pd.concat([df, dfWinsPolesY(y)])
    return df


def tabs():
    t04 = Constplot(2004)
    t05 = Constplot(2005)
    t06 = Constplot(2006)
    t07 = Constplot(2007)
    t08 = Constplot(2008)
    t09 = Constplot(2009)
    t10 = Constplot(2010)
    t11 = Constplot(2011)
    t12 = Constplot(2012)
    t13 = Constplot(2013)
    t14 = Constplot(2014)
    t15 = Constplot(2015)
    t16 = Constplot(2016)
    t17 = Constplot(2017)
    t18 = Constplot(2018)
    t19 = Constplot(2019)
    t20 = Constplot(2020)
    t21 = Constplot(2021)
    #t21a = [t21]
    tall = [t04,t05,t06,t07,t08,t09,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21]
    
    #tabs = Tabs(tabs= t21a)
    tabs = Tabs(tabs= tall[::-1])
        
    return tabs