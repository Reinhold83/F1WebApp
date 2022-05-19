import pandas as pd
import numpy as np
from copy import deepcopy
from bokeh.palettes import GnBu, RdPu, viridis
from math import pi
import json
import math


from bokeh.models import (ColumnDataSource, Select, HoverTool, FactorRange, Panel,Tabs, LabelSet, Label, StringFormatter,
                          NumeralTickFormatter, DatetimeTickFormatter,LinearInterpolator, Legend,GeoJSONDataSource,
                         ColorBar, LinearColorMapper, BasicTicker, PrintfTickFormatter, CustomJS,DataRange1d, CDSView,
                         BoxAnnotation, HArea)

from bokeh.models import LinearAxis, CategoricalAxis
from bokeh.models.tools import CustomJSHover
from bokeh.models.widgets import DataTable, TableColumn, HTMLTemplateFormatter, Div, Dropdown


from bokeh.transform import dodge, jitter, factor_cmap, cumsum
from bokeh.plotting import figure, save, reset_output
from bokeh.io import output_notebook, show, output_file, export_png
from bokeh.io import export_svgs
from bokeh.layouts import column, row, gridplot, grid
np.warnings.filterwarnings('ignore')


def dfPie(year):
    
    dfR = pd.read_csv('data/races.csv', index_col=0, delimiter=',')
    dfR = dfR[dfR['year'] == int(year)]
    dfR.drop('url', axis=1, inplace=True)
    dfR.sort_values('date', inplace=True)
    
    
    #df contructors teams
    dfC = pd.read_csv('data/constructors.csv', delimiter=',', index_col=0)
    
    #df constructor standings
    dfC_S = pd.read_csv('data/constructor_results.csv', delimiter=',', index_col=0)
    dfC_S.set_index('raceId', inplace=True)
    #dfC_S = pd.read_csv( 'data/constructor_standings.csv', index_col=1, delimiter=',')
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
    dfL = pd.DataFrame(dfC_S1.groupby(['Team','GP'], sort=False)['points'].sum())
    dfL['c'] = dfL.index.get_level_values('Team').map(diC).fillna('white')
      
    
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
    
    #dflaps to check laps lead
    dflaps = pd.read_csv('data/lap_times.csv', delimiter=',', index_col=0)
    dflaps = dflaps[dflaps.index.isin(CSF)]
    dflaps['DN'] = dflaps.driverId.map(diDidxDN)
    dflaps['Team'] = dflaps.DN.map(diDxT)
    
    #df driver standings champioship
    dfD_C = pd.read_csv('data/driver_standings.csv', sep=',', index_col='raceId')
    dfD_CF = dfD_C.index.isin(qF)
    dfD_C = dfD_C[dfD_CF]
    dfD_C['DN'] = dfD_C.driverId.map(diDidxDN)
    dfD_C['Team'] = dfD_C.DN.map(diDxT)
    
       
    #costructor cham
    dfC_S2 = dfC_S1.sort_values('points', na_position ='first')
    dfC_S2['cumsum'] = pd.DataFrame(dfC_S2.groupby('Team')['points'].cumsum())
    #dfC_S2.replace('Force India', 'Racing Point', inplace=True)
    #dfC_S2.replace('Renault', 'Alpine F1 Team', inplace=True)
    #dfC_S2.replace('Toro Rosso', 'AlphaTauri',inplace=True)
    dfC_S3 = pd.DataFrame(dfC_S2.groupby('Team')['points'].sum())
    dfC_S3.sort_values('points', ascending=False, inplace=True)
    dfC_S3['c'] = dfC_S3.index.map({'Red Bull': '#121F45', 'Ferrari':'#A6051A', 'Mercedes':'#00A19C',
                                       'Alpine F1 Team': '#005BA9', 'Haas F1 Team': '#F9F2F2', 'Williams': '#005AFF',
                                       'AlphaTauri': '#00293F','McLaren': '#FF8000', 'Alfa Romeo': '#981E32','Aston Martin':'#00352F',
                                   'Racing Point':'#F596C8', 'Renault':'#FFF500', 'Toro Rosso':'#469BFF', 'Catherham':'#048646',
                                   'Sauber':'#9B0000','Force India':'#F596C8','Lotus F1':'#FFB800', 'Marussia':'#6E0000',
                                   'Jaguar':'#08623e', 'Toyota':'#8e0018', 'BAR':'#d5d5d3', 'Jordan':'#ead136',
                                   'Prost':'#03033f', 'Super Aguri': '#b61211', 'Benetton':'#72dffe', 'Arrows':'#fc8d2c',
                                   'Honda':'#cccfc8', 'BMW Sauber':'#ffffff', 'Brawn':'#cccfc8','Minardi':'black'}).fillna('white')
    
    pos = ['1st', '2nd', '3rd', '4th', '5th','6th','7th','8th','9th','10th','11th','12th','13th','14th']   
    dfC_S3['pos'] = pos[:len(dfC_S3.index)]    
    wins = dfD_C[dfD_C.wins >= 1].groupby(['raceId', 'Team'])['wins'].sum().reset_index()
    dfC_S3['wins'] = wins[wins.raceId == wins.raceId.max()].set_index('Team').drop('raceId', axis=1)
    dfC_S3['wins'].fillna(0, inplace=True)
    dfC_S3['poles'] = dfQ1[dfQ1.position == 1].groupby('Team')['position'].count()
    dfC_S3.poles.fillna(0, inplace=True)
    dfC_S3['FastLap'] = dfRes[dfRes['rank'] == 1].groupby('Team')['rank'].sum()
    dfC_S3['FastLap'] = dfC_S3['FastLap'].fillna(0)
    dfC_S3['LapsLed'] = pd.DataFrame(dflaps[dflaps.position == 1].groupby('Team')['position'].sum())
    dfC_S3['LapsLed'] = dfC_S3['LapsLed'].fillna(0)
    dfC_S3.iloc[::,-4:] = dfC_S3.iloc[::,-4:].astype(int)
    dfC_S3['year'] = year
    dfC_S3 = dfC_S3.reset_index().set_index('year')
    if year == 2020:
        dfC_S3.iloc[0,-1] = dfC_S3.iloc[0,-1] + dfC_S3[dfC_S3.Team == 'Williams']['LapsLed'].values
        dfC_S3 = dfC_S3[dfC_S3.Team != 'Williams']
    
    return dfC_S3



def fw2(df, col):
    df = df.reset_index()
    df = df[['year','Team','c',str(col)]].sort_values(str(col), ascending=False)
    df = df[df[str(col)] >= 1]
    df['pct'] = round(((df[str(col)] / df[str(col)].sum())*100),2)
    df['angle'] = df[str(col)] / df[str(col)].sum() * pi
    
    return df

def allYears():
    df = dfPie(2004)
    for i in range(2005,2022):
        df = pd.concat([df, dfPie(i)])
        dfw = fw2(df, 'wins')
        dfw.to_csv('dfwins.csv')
        dfp = fw2(df, 'poles')
        dfp.to_csv('dfpoles.csv')
        dff = fw2(df, 'FastLap')
        dff.to_csv('dfFastLap.csv')
        dfl = fw2(df, 'LapsLed')
        dfl.to_csv('dfLapsLed.csv')
        df.to_csv('dfall.csv')
    return df


def df_build(year):
    
    dfR = pd.read_csv('data/races.csv', index_col=0, delimiter=',')
    dfR = dfR[dfR['year'] == int(year)]
    dfR.drop('url', axis=1, inplace=True)
    dfR.sort_values('date', inplace=True)
    dfR['name'] = dfR['name'].str.replace('Grand Prix', '')
    
    
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
    dfL = pd.DataFrame(dfC_S1.groupby(['Team','GP'], sort=False)['points'].sum())
    dfL['c'] = dfL.index.get_level_values('Team').map(diC).fillna('white')
    dfL['year'] = year
    #dfL.replace('Grand Prix', '', inplace=True)
    
    
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
    
    #dflaps to check laps lead
    dflaps = pd.read_csv('data/lap_times.csv', delimiter=',', index_col=0)
    dflaps = dflaps[dflaps.index.isin(CSF)]
    dflaps['DN'] = dflaps.driverId.map(diDidxDN)
    dflaps['Team'] = dflaps.DN.map(diDxT)
    
    
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
                                   'Honda':'#cccfc8', 'BMW Sauber':'#ffffff', 'Brawn':'#cccfc8','Minardi':'black'})
    
    pos = ['1st', '2nd', '3rd', '4th', '5th','6th','7th','8th','9th','10th','11th','12th','13th','14th']   
    dfC_S3['pos'] = pos[:len(dfC_S3.index)]    
    wins = dfD_C[dfD_C.wins >= 1].groupby(['raceId', 'Team'])['wins'].sum().reset_index()
    dfC_S3['wins'] = wins[wins.raceId == wins.raceId.max()].set_index('Team').drop('raceId', axis=1)
    dfC_S3['wins'].fillna(0, inplace=True)
    dfC_S3['poles'] = dfQ1[dfQ1.position == 1].groupby('Team')['position'].count()
    dfC_S3.poles.fillna(0, inplace=True)
    dfC_S3['FastLap'] = dfRes[dfRes['rank'] == 1].groupby('Team')['rank'].sum()
    dfC_S3['FastLap'] = dfC_S3['FastLap'].fillna(0)
    dfC_S3['LapsLed'] = pd.DataFrame(dflaps[dflaps.position == 1].groupby('Team')['position'].sum())
    dfC_S3['LapsLed'] = dfC_S3['LapsLed'].fillna(0)
    dfC_S3.iloc[::,-4:] = dfC_S3.iloc[::,-4:].astype(int)
    dfC_S3['year'] = year
    
    dfC_S3.to_csv(f'df{year}.csv')
    dfL.to_csv(f'dfL{year}.csv')



def df_exports():
    for i in range(2004, 2022):
        df_build(i)



def fw(df, col, year):
    df = df[df.year == year]
    df = df.reset_index()
    df = df[['year','Team','c',str(col)]].sort_values(str(col), ascending=False)
    df = df[df[str(col)] >= 1]
    df['pct'] = round(((df[str(col)] / df[str(col)].sum())*100),2)
    df['angle'] = df[str(col)] / df[str(col)].sum() * pi
    
    return df


def LPlot(df, year):
    
    #dict colors x teams
    diC = {'Red Bull': '#121F45', 'Ferrari':'#A6051A', 'Mercedes':'#00A19C','Alpine F1 Team': '#005BA9', 'Haas F1 Team': '#F9F2F2', 'Williams': '#005AFF',
            'AlphaTauri': '#00293F','McLaren': '#FF8000', 'Alfa Romeo': '#981E32','Aston Martin':'#00352F','Racing Point':'#F596C8','Renault':'#FFF500',
           'Toro Rosso':'#469BFF', 'Catherham':'#048646', 'Sauber':'#9B0000','Force India':'#F596C8','Lotus F1':'#FFB800', 'Marussia':'#6E0000',
            'Jaguar':'#08623e', 'Toyota':'#8e0018', 'BAR':'#d5d5d3', 'Jordan':'#ead136','Prost':'#03033f', 'Super Aguri': '#b61211', 'Benetton':'#72dffe', 'Arrows':'#fc8d2c',
            'Honda':'#cccfc8', 'BMW Sauber':'#ffffff', 'Brawn':'#cccfc8', 'Lotus':'#FFB800'}

    d = []
    gps = df.sort_values('points', ascending=True)['GP'].unique()
    cl = pd.Series(df.index.unique().values, range(0, len(df.index.unique()))).to_dict()
    
    for p in df.groupby('Team', sort=False)['points']:
        np.array(d.append(p[1].values)).flatten()
        df1 = pd.DataFrame(d, columns=gps)
        df1 = df1.T
        df1.fillna(0, inplace=True)
        df1.rename(columns = cl, inplace = True)
        df1 = df1.cumsum()
        
    p1 = figure(x_range=np.array(df1.index), title='Constructor Championship by Grand Prix '+ str(year), plot_height=400, plot_width=800, y_axis_label = 'points',
                 tools='pan, wheel_zoom, box_zoom, reset', toolbar_location='right')
    p1.grid.grid_line_alpha = .25
    p1.grid.grid_line_dash = 'dotted'
    p1.grid.grid_line_dash_offset = 5
    p1.grid.grid_line_width = 2
    p1.grid.grid_line_color = 'white'
    p1.axis.major_label_text_font_style = 'bold'
    p1.xaxis.major_label_text_font_size = '4.5px'
    p1.yaxis.major_label_text_font_size = '11px'
    p1.title.text_font_size = '13.5px'
    p1.title.text_color = 'white'
    p1.outline_line_color=None
    p1.yaxis.axis_label_text_color= 'white'
    p1.axis.major_label_text_color= 'white'
    p1.y_range=DataRange1d(only_visible=True)
   
    #p1.yaxis.ticker.desired_num_ticks = 12
    p1.toolbar.autohide = True

    p1.background_fill_color = '#010727'
    p1.background_fill_alpha =.8
    p1.border_fill_color =  '#010727'
    p1.border_fill_alpha = .8
    p1.axis.axis_line_color ='white'
    p1.axis.minor_tick_line_color ='white'
    p1.axis.major_tick_line_color ='white'
    p1.x_range.range_padding = -.02
    
    gps1 = pd.DataFrame(df1.iloc[-1]).reset_index()
    gps1.columns = ['Teams','points']
    gps1 = gps1.sort_values('points', ascending=False)
    gps1['c'] = gps1.Teams.map(diC).fillna('white')
    color = np.array(gps1.c)
    gps1 = np.array(gps1['Teams'])
    xh = np.array(df1.index)
    #legend = Legend()
    #p1.add_layout(legend, 'below')

    
    for col,c in zip(gps1, color):
        line = p1.line(df1.index, df1[col], color=c, line_width=3.5, line_join = 'bevel', legend_label=col)
        p1.circle(df1.index, df1[col], color=c, line_color='white', line_width=.15, size=6.5, legend_label=col)
        
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
        p1.legend.label_text_font_size = '5.5px'
        p1.legend.click_policy='hide'
        p1.y_range=DataRange1d(only_visible=True)
        #hv = HoverTool(tooltips=[('Team', '@col'), ('Points', '@col{0.0}')], renderers=[line])
        #p1.add_tools(hv)
        #p1.y_range.start = -10
        #p1.y_range.end = df1.values.max() * 1.1

        #p1.add_tools(HoverTool( tooltips=[('GP', '@x{custom}'), ('Team', '@y{custom}')], formatters=dict(x=x_custom, y=y_custom)))

    return p1

def TeamxTeam(dfL, year):
    d = []
    gps = dfL.sort_values('points', ascending=True)['GP'].unique()
    cl = pd.Series(dfL.index.unique().values, range(0, len(dfL.index.unique()))).to_dict()
    dicolors = dfL.c.to_dict()

    for p in dfL.groupby('Team', sort=False)['points']:
        np.array(d.append(p[1].values)).flatten()
        df1 = pd.DataFrame(d, columns=gps)
        df1 = df1.T
        df1.fillna(0, inplace=True)
        df1.rename(columns = cl, inplace = True)
        df1 = df1.cumsum()

    df1['Ferrari_'] = df1['Ferrari']
    df1['McLaren_'] = df1['McLaren']
    df1.index = df1.index.str[:-1]
    df1.index.name = 'GPS'

    src1 = ColumnDataSource(df1)

    ptl = figure(x_range=FactorRange(*np.array(df1.index)), plot_height=350, plot_width=800, title='Ferrari x McLaren ' + str(year),
                tools='pan, wheel_zoom, box_zoom, reset', toolbar_location='right', y_axis_label='points')
    t1g = ptl.vbar(x=dodge('GPS', -.3, range=ptl.x_range), top='Ferrari_', width=.25, color=dicolors['Ferrari'], line_color='white', 
             line_width=.3, source=src1, hatch_alpha=1.0, hover_fill_alpha=0.7, hover_fill_color='#b84e04')

    t2g = ptl.vbar(x=dodge('GPS', 0, range=ptl.x_range), top='McLaren_', width=.25, fill_color=dicolors['McLaren'],
             line_width=.3, line_color='white', source=src1, hatch_alpha=1.0, hover_fill_alpha=0.7, hover_fill_color='#7851a9')

    ptl.title.text_font_size = '9px'
    ptl.grid.grid_line_alpha = .25
    ptl.grid.grid_line_dash = 'dotted'
    ptl.grid.grid_line_dash_offset = 5
    ptl.grid.grid_line_width = 2
    ptl.grid.grid_line_color = 'white'
    ptl.axis.major_label_text_font_style = 'bold'
    ptl.xaxis.major_label_text_font_size = '5.5px'
    ptl.yaxis.major_label_text_font_size = '11px'
    ptl.title.text_font_size = '15px'
    ptl.title.text_color = 'white'
    ptl.outline_line_color=None
    ptl.yaxis.axis_label_text_color= 'white'
    ptl.axis.major_label_text_color= 'white'
    ptl.toolbar.autohide = True
    ptl.outline_line_color=None

    ptl.background_fill_color = '#010727'
    ptl.background_fill_alpha =.8
    ptl.border_fill_color =  '#010727'
    ptl.border_fill_alpha = .8
    ptl.axis.axis_line_color ='white'
    ptl.axis.minor_tick_line_color ='white'
    ptl.axis.major_tick_line_color ='white'
    ptl.y_range.start = 0

    opts = df1.columns[:-2].to_list()
    #opts.insert(0, 'Ferrari_')


    hl1 = HoverTool(renderers=[t1g], tooltips=({'Grand Prix': '@GPS', 'Points': '@Ferrari_{0.0}'}), point_policy='follow_mouse')
    ptl.add_tools(hl1)

    hl2 = HoverTool(renderers=[t2g], tooltips=({'Grand Prix': '@GPS', 'Points': '@McLaren_{0.0}'}), point_policy='follow_mouse')
    ptl.add_tools(hl2)


    select1 = Select(title='Select Team A', align='start', value='Ferrari_', width=115, height=15, options=opts)

    select2 = Select(title='Select Team B', align='start', value='McLaren_', width=115, height=15, options=opts)

    select1.js_on_change('value', CustomJS(args=dict(source=src1, select=select1, dic=dicolors, p=t1g, title=ptl.title,
                                                   select1=select2, year=year), code="""
        const new_data = Object.assign({}, source.data)
        new_data.Ferrari_ = source.data[select.value]
        source.data = new_data
        p.glyph.fill_color = dic[select.value]

        title.text = select.value + ' x '+ select1.value + ' ' + year


    """))

    select2.js_on_change('value', CustomJS(args=dict(source=src1, select=select2, select1=select1, dic=dicolors,
                                                     p=t2g, title=ptl.title, year=year), code="""
        const new_data = Object.assign({}, source.data)
        new_data.McLaren_ = source.data[select.value]
        source.data = new_data
        p.glyph.fill_color = dic[select.value]

        title.text = select1.value + ' x '+ select.value + ' ' + year

    """))

    ctl =  column([row([select1,select2]), ptl], spacing=30)
    
    return ctl
    
def Constplot(year):
   
    df = pd.read_csv(f'data/df{year}.csv', delimiter=',', index_col=0)
    srcC = ColumnDataSource(df)
    xC = np.array(df.index.unique())

    pCons = figure(x_range=xC, title='Constructor Championship ' + str(year), plot_height=400, plot_width=800, y_axis_label = 'points',
                     tools='pan, wheel_zoom, box_zoom, reset', toolbar_location='right')
    pCons.vbar(top='points', width=.6, x='Team',  source=srcC, line_color='white', line_width = .2, fill_color='c',
              hatch_alpha=1.0, hover_fill_alpha=0.7, hover_fill_color='#FEE715')

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
    
    
    
    
    
    dfw = fw(df, 'wins', year)
    sw = ColumnDataSource(dfw)
    

    pw = figure(plot_height=150, plot_width=200,toolbar_location='below', tools = 'pan, wheel_zoom, box_zoom, reset',
                 x_range=(-1.05, 1.05), y_range=(.6,.3), x_axis_type=None, y_axis_type=None)
    pw_ = pw.wedge(x=0, y=.5, radius=1, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='white', fill_color='c', line_width=.2,  source=sw, hatch_alpha=1.0, hover_fill_alpha=0.7, hover_fill_color='#FEE715')
    pw.add_tools(HoverTool(renderers=[pw_], tooltips=[('Team','@Team'), ('Wins','@wins'), ('','@pct{0.0} %'), ('year',str(year))],point_policy='follow_mouse'))
    #white
    pw.wedge(x=0, y=.505, radius=.6, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='#010727', fill_color='#010727', fill_alpha=1.0, source=sw)
    
    lw = Label(x=0, y=.49, x_offset=0, y_offset=0, text='Wins',
           text_color='white', text_font_size='15px', text_font_style='bold', text_align='center')
    pw.add_layout(lw)
    pw.outline_line_color=None
    pw.background_fill_color = '#010727'
    pw.background_fill_alpha =.8
    pw.border_fill_color =  '#010727'
    pw.border_fill_alpha = .8


    
    
    dfp = fw(df, 'poles', year)
    sp = ColumnDataSource(dfp)
    pp = figure(plot_height=150, plot_width=200,toolbar_location='below', tools = 'pan, wheel_zoom, box_zoom, reset',
                 x_range=(-1.05, 1.05), y_range=(.6,.3), x_axis_type=None, y_axis_type=None)
    pp_ = pp.wedge(x=0, y=.5, radius=1, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='white', fill_color='c', line_width=.2, source=sp, hatch_alpha=1.0, hover_fill_alpha=0.7, hover_fill_color='#FEE715')
    pp.add_tools(HoverTool(renderers=[pp_], tooltips=[('Team','@Team'), ('Poles','@poles'), ('','@pct{0.0} %'), ('year',str(year))],point_policy='follow_mouse'))
    
    #white
    pp.wedge(x=0, y=.505, radius=.6, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='#010727', fill_color='#010727', source=sp)
    
    lp = Label(x=0, y=.49, x_offset=0, y_offset=0, text='Poles',
           text_color='white', text_font_size='15px', text_font_style='bold', text_align='center')
    pp.add_layout(lp)

    pp.outline_line_color=None
    pp.background_fill_color = '#010727'
    pp.background_fill_alpha =.8
    pp.border_fill_color =  '#010727'
    pp.border_fill_alpha = .8
    
    
    dff = fw(df, 'FastLap', year)
    sf = ColumnDataSource(dff)    
    pf = figure(plot_height=150, plot_width=200,toolbar_location='below', tools = 'pan, wheel_zoom, box_zoom, reset',
                 x_range=(-1.05, 1.05), y_range=(.6,.3), x_axis_type=None, y_axis_type=None)   
    pf_ = pf.wedge(x=0, y=.5, radius=1, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='white', fill_color='c', line_width=.2,  source=sf, hatch_alpha=1.0, hover_fill_alpha=0.7, hover_fill_color='#FEE715')
    pf.add_tools(HoverTool(renderers=[pf_], tooltips=[('Team','@Team'), ('Fast laps','@FastLap'), ('','@pct{0.0} %'), ('year',str(year))],point_policy='follow_mouse'))   
    #white
    pf.wedge(x=0, y=.505, radius=.6, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='#010727', fill_color='#010727', fill_alpha=1.0, source=sf)
    
    lf = Label(x=0, y=.49, x_offset=0, y_offset=0, text='Fastest laps',
           text_color='white', text_font_size='12px', text_font_style='bold', text_align='center')
    pf.add_layout(lf)
    pf.outline_line_color=None
    pf.toolbar.autohide = True
    pf.background_fill_color = '#010727'
    pf.background_fill_alpha =.8
    pf.border_fill_color =  '#010727'
    pf.border_fill_alpha = .8


    
    
    dfl = fw(df, 'LapsLed', year)
    sl = ColumnDataSource(dfl)
    pl = figure(plot_height=150, plot_width=200,toolbar_location='below', tools = 'pan, wheel_zoom, box_zoom, reset',
                 x_range=(-1.05, 1.05), y_range=(.6,.3), x_axis_type=None, y_axis_type=None)
    pl_ = pl.wedge(x=0, y=.5, radius=1, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='white', fill_color='c', line_width=.2,  source=sl, hatch_alpha=1.0, hover_fill_alpha=0.7, hover_fill_color='#FEE715')
    pl.add_tools(HoverTool(renderers=[pl_], tooltips=[('Team','@Team'), ('Laps led','@LapsLed'), ('','@pct{0.0} %'), ('year',str(year))],point_policy='follow_mouse'))
    #white
    pl.wedge(x=0, y=.505, radius=.6, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='#010727', fill_color='#010727', fill_alpha=1.0, source=sl)
    
    ld = Label(x=0, y=.49, x_offset=0, y_offset=0, text='Laps led',
           text_color='white', text_font_size='13px', text_font_style='bold', text_align='center')
    pl.add_layout(ld)
    pl.outline_line_color=None
    pl.toolbar.autohide = True
    pl.background_fill_color = '#010727'
    pl.background_fill_alpha =.8
    pl.border_fill_color =  '#010727'
    pl.border_fill_alpha = .8    
    
    #r = row([pw, pp, pf, pl], spacing=-11, align='end')
    r = gridplot([[pw, pp, pf, pl]], merge_tools=True, toolbar_location='right', width=200, height=150)


    dfL = pd.read_csv(f'data/dfL{year}.csv', delimiter=',', index_col=0)
    pL = LPlot(dfL, year)
    
    txt = TeamxTeam(dfL, year)
    
    t1 = Panel(child=pCons, title='Overall')
    t2 = Panel(child =pL, title='By Grand Prix')
    tabs = Tabs(tabs= [t1, t2])
    
    col = column([tabs, txt, r], align='start')
    
    t = Panel(child=col)
    
    
    return t


def tabs():  
    
    Campionship_2021, Campionship_2020,Campionship_2019, Campionship_2018, Campionship_2017,Campionship_2016  = Constplot(2021), Constplot(2020), Constplot(2019), Constplot(2018), Constplot(2017), Constplot(2016)
    Campionship_2015,Campionship_2014,Campionship_2013,Campionship_2012,Campionship_2011,Campionship_2010 = Constplot(2015), Constplot(2014), Constplot(2013), Constplot(2012), Constplot(2011),Constplot(2010)
    Campionship_2009,Campionship_2008,Campionship_2007,Campionship_2006,Campionship_2005,Campionship_2004 = Constplot(2009), Constplot(2008), Constplot(2007), Constplot(2006), Constplot(2005), Constplot(2004)

    tall = [Campionship_2021, Campionship_2020,Campionship_2019, Campionship_2018, Campionship_2017,Campionship_2016,
    Campionship_2015,Campionship_2014,Campionship_2013,Campionship_2012,Campionship_2011,Campionship_2010,
    Campionship_2009,Campionship_2008,Campionship_2007,Campionship_2006,Campionship_2005,Campionship_2004]

    ops=['Campionship_2021', 'Campionship_2020','Campionship_2019', 'Campionship_2018','Campionship_2017','Campionship_2016',
             'Campionship_2015','Campionship_2014', 'Campionship_2013','Campionship_2012', 'Campionship_2011','Campionship_2010',
             'Campionship_2009','Campionship_2008', 'Campionship_2007','Campionship_2006','Campionship_2005','Campionship_2004']

    diPanel = {ops[i]: tall[i] for i in range(len(tall))}

    select = Select(title='Select year:', align='start', value='Campionship_2021', options=ops, width=150, margin = (5, 5, 0, 0))
    tabs = Tabs(tabs= tall)
    col = column([select, tabs])

    select.js_on_change('value', CustomJS(args=dict(sel=select,tab=tabs,diPanel=diPanel)
              ,code='''
              var sv = sel.value
              tab.tabs = [diPanel[sv]]
              '''))
    return col




dfall = pd.read_csv('data/dfall.csv', delimiter=',', index_col=0)
dfall#[dfall.Team == 'Mercedes']

ferrari = dfall[dfall.Team == 'Ferrari']
redbull = dfall[dfall.Team.isin(['Jaguar','Red Bull'])]
mercedes = dfall[dfall.Team.isin(['BAR','Honda', 'Brawn', 'Mercedes'])]
mclaren = dfall[dfall.Team == 'McLaren']
alfaromeo = dfall[dfall.Team.isin(['BMW Sauber', 'Sauber', 'Alfa Romeo'])]
alpine = dfall[dfall.Team.isin(['Renault','Lotus F1','Alpine F1 Team'])]
williams = dfall[dfall.Team == 'Williams']
astonmartin = dfall[dfall.Team.isin(['MF1','Jordan','Spyker','Force India','Racing Point','Aston Martin'])]
haas = dfall[dfall.Team.isin(['Virgin','Marussia','Virgin','Haas F1 Team'])]
alphatauri = dfall[dfall.Team.isin(['Minardi','Toro Rosso','AlphaTauri'])]
toyota = dfall[dfall.Team == 'Toyota']
#caterham = dfall[dfall.Team.isin(['Lotus','Caterham'])]



def yearlyPlots(df):
    df.index = df.index.astype(str)
    xC = np.array(df.index.unique())
    srcH = ColumnDataSource(df)
    
    pH= figure(x_range=xC, title=str(df.Team[-1]) + ' Constructor points by year', plot_height=300, plot_width=600, y_axis_label = 'points',
                     tools='pan, wheel_zoom, box_zoom, reset', toolbar_location='right')#, x_axis_type='datetime')

    pHb = BoxAnnotation(right=0, left=5.8, fill_alpha=0.15, fill_color='orange', line_color=None, left_units='data', right_units='data')
    pH.add_layout(pHb)

    pH.vbar(top='points', width=.6, x='year',  source=srcH, line_color='white', line_width = .2, fill_color='c',
           hatch_alpha=1.0, hover_fill_alpha=0.7, hover_fill_color='#FEE715')

    pH.grid.grid_line_alpha = .25
    pH.grid.grid_line_dash = 'dotted'
    pH.grid.grid_line_dash_offset = 5
    pH.grid.grid_line_width = 2
    pH.grid.grid_line_color = 'white'
    pH.axis.major_label_text_font_style = 'bold'
    pH.xaxis.major_label_text_font_size = '9.5px'
    pH.yaxis.major_label_text_font_size = '11px'
    pH.title.text_font_size = '13px'
    pH.title.text_color = 'white'
    pH.outline_line_color=None
    pH.yaxis.axis_label_text_color= 'white'
    pH.axis.major_label_text_color= 'white'
    pH.toolbar.autohide = True
    pH.y_range.start = -1

    pH.background_fill_color = '#010727'
    pH.background_fill_alpha =.8
    pH.border_fill_color =  '#010727'
    pH.border_fill_alpha = .8
    pH.axis.axis_line_color ='white'
    pH.axis.minor_tick_line_color ='white'
    pH.axis.major_tick_line_color ='white'
    hv = HoverTool()
    hv.tooltips=[('Team name', '@Team'), ('Points', '@points{0.0}'), ('Position', '@pos'), ('Year','@year')]
    pH.add_tools(hv)
    
    
    v1, v2 = df.iloc[:int(len(df.index)/2),-3:-1].values.sum(), df.iloc[int(len(df.index)/2):,-4:-1].values.sum()

    pl = []
    if v1 > v2:
        pl.append('top_right')
    else:
        pl.append('top_left')
    
    pO = figure(x_range=xC, title='Wins/Poles/Fastest Laps by year', plot_height=300, plot_width=600, y_axis_label = 'quantity',
                     tools='pan, wheel_zoom, box_zoom, reset', toolbar_location='below', x_axis_location = None, title_location='below')
    
    pO.line(x='year', y='wins', line_width=2.5, line_color='#3EB489', source=srcH, legend_label='Wins')
    pO.circle(x='year', y='wins', line_color='black', line_width=.3, color='#3EB489', size=7.5, source=srcH,
              legend_label='Wins', hatch_alpha=1.0, hover_fill_alpha=0.7, hover_fill_color='#FEE715')

    pO.line(x='year', y='poles', line_width=2.5, line_color='#7851a9', source=srcH, legend_label='Poles')
    pO.circle(x='year', y='poles', line_color='black', line_width=.3, color='#7851a9', size=7.5, source=srcH, 
              legend_label='Poles', hatch_alpha=1.0, hover_fill_alpha=0.7, hover_fill_color='#FEE715')

    
    pO.line(x='year', y='FastLap', line_width=2.5, line_color='#FEE715', source=srcH, legend_label='Fastest Laps')
    pO.circle(x='year', y='FastLap', line_color='black', line_width=.3, color='#FEE715', size=7.5, source=srcH,
              legend_label='Fastest Laps', hatch_alpha=1.0, hover_fill_alpha=0.7, hover_fill_color='#FEE715')

    pO.grid.grid_line_alpha = .25
    pO.grid.grid_line_dash = 'dotted'
    pO.grid.grid_line_dash_offset = 5
    pO.grid.grid_line_width = 2
    pO.grid.grid_line_color = 'white'
    pO.axis.major_label_text_font_style = 'bold'
    pO.xaxis.major_label_text_font_size = '9.5px'
    pO.yaxis.major_label_text_font_size = '11px'
    pO.title.text_font_size = '13px'
    pO.title.text_color = 'white'
    pO.outline_line_color=None
    pO.yaxis.axis_label_text_color= 'white'
    pO.axis.major_label_text_color= 'white'
    #pO.toolbar.autohide = True
    pO.y_range.start = 0
    pO.outline_line_color=None
    pO.legend.location= pl[0] #'top_right'#(370,180)
    pO.legend.background_fill_alpha= 0
    pO.legend.border_line_color = None
    pO.legend.label_text_color='white'
    pO.legend.label_text_font_size='9px'
    pO.legend.click_policy="hide"
    #pO.legend.title='↓ Disable/Enable'
    pO.legend.title_text_color='white'
    pO.legend.title_text_font_size='6px'
    pO.legend.orientation='horizontal'

    pO.background_fill_color = '#010727'
    pO.background_fill_alpha =.8
    pO.border_fill_color =  '#010727'
    pO.border_fill_alpha = .8
    pO.axis.axis_line_color ='white'
    pO.axis.minor_tick_line_color ='white'
    pO.axis.major_tick_line_color ='white'
    pO.y_range=DataRange1d(only_visible=True)
    hv1 = HoverTool()
    hv1.tooltips=[('Team name', '@Team'), ('Year','@year'), ('Wins', '@wins'), ('Poles', '@poles'), ('Fastest Laps', '@FastLap') ]
    pO.add_tools(hv1)
    
    
    x1 = np.zeros(len(df.index))
    x2 = np.array(df['pos'].str.extract('(\d+)'))[::-1]
    y = np.array(df.index) [::-1]
    x3 = np.array(df.pos)[::-1]
    c1 = np.array(df.c)[::-1]
    t1 = np.array(df.Team)[::-1]

    src = ColumnDataSource(data=dict(x1=x1, x2=x2, x3=x3, c=c1, t=t1, y=y))
    
    
    pA = figure(y_range=xC[::-1], title='Championship Position', plot_height=600, plot_width=200,
                     tools='pan, wheel_zoom, box_zoom, reset', toolbar_location='left')
    g = HArea(x1='x1', x2='x2', y='y', fill_color=df['c'][-1], hatch_color='orange' )
    pA.circle(x='x2', y='y', line_color='white', line_width=.4, color='c', size=5, source=src)

    pA.add_glyph(src, g)
    
    pA.title.text_font_size = '10px'
    pA.grid.grid_line_alpha = .25
    pA.grid.grid_line_dash = 'dotted'
    pA.grid.grid_line_dash_offset = 5
    pA.grid.grid_line_width = 2
    pA.grid.grid_line_color = 'white'
    pA.axis.major_label_text_font_style = 'bold'
    pA.xaxis.major_label_text_font_size = '9.5px'
    pA.yaxis.major_label_text_font_size = '11px'
    pA.title.text_color = 'white'
    pA.outline_line_color=None
    pA.yaxis.axis_label_text_color= 'white'
    pA.axis.major_label_text_color= 'white'
    pA.toolbar.autohide = True
    pA.x_range.start = 0.5
    pA.outline_line_color=None

    pA.background_fill_color = '#010727'
    pA.background_fill_alpha =.8
    pA.border_fill_color =  '#010727'
    pA.border_fill_alpha = .8
    pA.axis.axis_line_color ='white'
    pA.axis.minor_tick_line_color ='white'
    pA.axis.major_tick_line_color ='white'
    pA.xaxis[0].formatter = NumeralTickFormatter(format='0o')
    pA.y_range.range_padding = -.04
    hv2 = HoverTool()
    hv2.tooltips=[('Year','@y'), ('Team name', '@t'), ('Championship pos', '@x3') ]
    pA.add_tools(hv2)

    col = row([pA, column([pH, pO])])  
    #col = gridplot([[pA, pH, pO]], merge_tools=True, toolbar_location='below')
    t = Panel(child=col)  
    
    return t


def c2():
    
    Ferrari, Red_Bull, Mercedes  = yearlyPlots(ferrari), yearlyPlots(redbull), yearlyPlots(mercedes)
    McLaren, Alfa_Romeo, Alpine, Williams = yearlyPlots(mclaren), yearlyPlots(alfaromeo), yearlyPlots(alpine), yearlyPlots(williams)
    Aston_Martin, HAAS, AlphaTauri, Toyota = yearlyPlots(astonmartin), yearlyPlots(haas), yearlyPlots(alphatauri), yearlyPlots(toyota) 
    
    dfs= [ferrari, redbull, mercedes, mclaren, alfaromeo, alpine, williams, astonmartin, haas, alphatauri, toyota, ferrari, mclaren]
    teamsN = ['Ferrari', 'Red_Bull', 'Mercedes', 'McLaren', 'Alfa_Romeo', 'Alpine', 'Williams', 'Aston_Martin', 'HAAS', 'AlphaTauri', 'Toyota','Ferrari_', 'McLaren_']

    CDSs = {t: ColumnDataSource(c) for t, c in zip(teamsN, dfs)}

    pO = figure(x_range=np.array(ferrari.index), title='Teams 2004-2021', plot_height=400, plot_width=770, y_axis_label = 'Points',
                     tools='pan, wheel_zoom, box_zoom, reset', toolbar_location='below', title_location='above')

    pOb = BoxAnnotation(right=0, left=5.8, fill_alpha=0.15, fill_color='orange', line_color=None, left_units='data', right_units='data')
    pO.add_layout(pOb)

    diRenders1 = {'Ferrari' : pO.line(x='year', y='points', line_width=4, line_color='#A6051A', source=CDSs['Ferrari'], legend_label='Ferrari'),
    'McLaren' : pO.line(x='year', y='points', line_width=4, line_color='#FF8000', source=CDSs['McLaren'], legend_label='McLaren'),
    'Red Bull' : pO.line(x='year', y='points', line_width=4, line_color='#121F45', source=CDSs['Red_Bull'], legend_label='Red Bull'),
    'Mercedes' : pO.line(x='year', y='points', line_width=4, line_color='#00A19C', source=CDSs['Mercedes'], legend_label='Mercedes'),
    'Alfa Romeo' : pO.line(x='year', y='points', line_width=4, line_color='#981E32', source=CDSs['Alfa_Romeo'], legend_label='Alfa Romeo'),
    'Alpine' : pO.line(x='year', y='points', line_width=4, line_color='#005BA9', source=CDSs['Alpine'], legend_label='Alpine'),
    'AlphaTauri' : pO.line(x='year', y='points', line_width=4, line_color='#00293F', source=CDSs['AlphaTauri'], legend_label='AlphaTauri'),
    'Williams' : pO.line(x='year', y='points', line_width=4, line_color='#005AFF', source=CDSs['Williams'], legend_label='Williams'),
    'Aston Martin' : pO.line(x='year', y='points', line_width=4, line_color='#00352F', source=CDSs['Aston_Martin'], legend_label='Aston Martin'),
    'HAAS' : pO.line(x='year', y='points', line_width=4, line_color='#F9F2F2', source=CDSs['HAAS'], legend_label='HAAS'),
    'Toyota' : pO.line(x='year', y='points', line_width=4, line_color='#8e0018', source=CDSs['Toyota'], legend_label='Toyota')}

    pO.add_layout(pO.legend[0], 'below')
    pO.grid.grid_line_alpha = .25
    pO.grid.grid_line_dash = 'dotted'
    pO.grid.grid_line_dash_offset = 5
    pO.grid.grid_line_width = 2
    pO.grid.grid_line_color = 'white'
    pO.axis.major_label_text_font_style = 'bold'
    pO.xaxis.major_label_text_font_size = '9.5px'
    pO.yaxis.major_label_text_font_size = '11px'
    pO.title.text_font_size = '13px'
    pO.title.text_color = 'white'
    pO.outline_line_color=None
    pO.yaxis.axis_label_text_color= 'white'
    pO.axis.major_label_text_color= 'white'
    #pO.toolbar.autohide = True
    pO.y_range.start = 0
    pO.outline_line_color=None
    pO.legend.location= 'top_left' #'top_right'#(370,180)
    pO.legend.background_fill_alpha= 0
    pO.legend.border_line_color = None
    pO.legend.label_text_color='white'
    pO.legend.label_text_font_size='9px'
    pO.legend.click_policy="hide"
    #pO.legend.title='↓ Disable/Enable'
    pO.legend.title_text_color='white'
    pO.legend.title_text_font_size='6px'
    pO.legend.orientation='horizontal'
    pO.legend.inactive_fill_alpha = .4
    pO.legend.inactive_fill_color = 'orange'
    pO.background_fill_color = '#010727'
    pO.background_fill_alpha =.8
    pO.border_fill_color =  '#010727'
    pO.border_fill_alpha = .8
    pO.axis.axis_line_color ='white'
    pO.axis.minor_tick_line_color ='white'
    pO.axis.major_tick_line_color ='white'
    pO.x_range.range_padding = -.03
    pO.y_range=DataRange1d(only_visible=True)
    hv1 = HoverTool()
    hv1.tooltips=[('Team name', '@Team'), ('Year','@year'), ('Champ. pos.', '@pos'), ('Points', '@points{0.0}'),
                  ('Wins', '@wins'), ('Poles', '@poles'), ('Fastest Laps', '@FastLap'), ('Laps led', '@LapsLed') ]
    pO.add_tools(hv1)



    select = Select(title='Select team A:', align='start', value='Ferrari', options=list(diRenders1.keys()), width=130, margin = (15, 5, 0, 0))
    select1 = Select(title='Select team B:', align='start', value='Mercedes', options=list(diRenders1.keys()), width=130, margin = (15, 5, 0, 0))

    cb = CustomJS(args=dict(sel=select, sel1=select1, rend_dict=diRenders1, title= pO.title),code='''
                  //get the value of the select
                  var sv = sel.value
                  var sv1 = sel1.value
                  for (var [k,v] of Object.entries(rend_dict)){
                          if (k==sv || k==sv1){
                                  v.visible=true}
                          else {
                              v.visible=false}}
                   title.text = sel.value + ' x '+ sel1.value + ' 2004-2021'
                  ''')
    select.js_on_change('value',cb)

    cb1 = CustomJS(args=dict(sel=select1, sel1=select, rend_dict=diRenders1, title= pO.title),code='''
                  //get the value of the select
                  var sv = sel.value
                  var sv1 = sel1.value
                  for (var [k,v] of Object.entries(rend_dict)){
                          if (k==sv || k==sv1){
                                  v.visible=true
                                  }
                          else {
                              v.visible=false}}
                  title.text = sel1.value + ' x '+ sel.value + ' 2004-2021'
                  ''')
    select1.js_on_change('value',cb1)


    ct = column([row([select, select1]), pO])
    
    tall = [Ferrari, Red_Bull, Mercedes, McLaren, Alfa_Romeo, Alpine, Williams, Aston_Martin, HAAS, AlphaTauri,Toyota]

    ops=['Ferrari', 'Red_Bull', 'Mercedes', 'McLaren', 'Alfa_Romeo', 'Alpine', 'Williams', 'Aston_Martin', 'HAAS', 'AlphaTauri','Toyota']

    diPanel = {ops[i]: tall[i] for i in range(len(tall))}

    sel_ = Select(title='Select team:', align='start', value='Ferrari', options=ops, width=150, margin = (5, 5, 0, 0))
    tabs = Tabs(tabs= tall)
    #col = column([select, tabs, pit], spacing=-10)
    col = column([sel_, tabs, ct])

    sel_.js_on_change('value', CustomJS(args=dict(sel=sel_,tab=tabs,diPanel=diPanel)
              ,code='''
              var sv = sel.value
              tab.tabs = [diPanel[sv]]
              '''))
    return col

def cols2():
    pit = figure(plot_width=780, plot_height=140, frame_height=145, frame_width=790, tools='', x_axis_location = None, y_axis_location = None)
    pit.image_url(url=['pointsystemTable.png'], x=0, y=0, w=740, w_units='screen', h=140, h_units='screen', anchor='center' )
    pit.background_fill_color = '#000626f6'
    pit.background_fill_alpha =.8
    pit.border_fill_color =  '#000626f6'
    pit.border_fill_alpha = .8
    pit.grid.grid_line_color=None
    pit.outline_line_color=None
    pit.toolbar_location = None
    
    c1, c2a = tabs(), c2()
    c = column([c1, c2a], align='center', spacing=-20)
    c3 =  column([c, pit], align='center')
    
    return c3

