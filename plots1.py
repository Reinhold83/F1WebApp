
import pandas as pd
import numpy as np
from copy import deepcopy
from bokeh.palettes import GnBu, RdPu, viridis
from math import pi
import json
import math


from bokeh.models import (ColumnDataSource, Select, HoverTool, FactorRange, Panel,Tabs, LabelSet, Label, StringFormatter,
                          NumeralTickFormatter, DatetimeTickFormatter,LinearInterpolator, Legend,GeoJSONDataSource,
                         ColorBar, LinearColorMapper, BasicTicker, PrintfTickFormatter, CustomJS,DataRange1d)

from bokeh.models import LinearAxis, CategoricalAxis
from bokeh.models.tools import CustomJSHover


from bokeh.transform import dodge, jitter, factor_cmap, cumsum
from bokeh.plotting import figure, save, reset_output
#import matplotlib.pyplot as plt
from bokeh.io import output_notebook, show, output_file, export_png
from bokeh.io import export_svgs
from bokeh.layouts import column, row, gridplot, grid


np.warnings.filterwarnings('ignore')


def fw(df, col):
    df = df[['c',str(col)]].sort_values(str(col), ascending=False)
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
    gps = df.sort_values('points', ascending=True).index.get_level_values('GP').unique()
    cl = pd.Series(df.index.get_level_values('Team').unique().values, range(0, len(df.index.get_level_values('Team').unique()))).to_dict()
    
    for p in df.groupby('Team', sort=False)['points']:
        np.array(d.append(p[1].values)).flatten()
        df1 = pd.DataFrame(d, columns=gps)
        df1 = df1.T
        df1.fillna(0, inplace=True)
        df1.rename(columns = cl, inplace = True)
        df1 = df1.cumsum()
        
    p1 = figure(x_range=np.array(df1.index), title='Constructor Championship by Grand Prix '+ str(year), plot_height=400, plot_width=750, y_axis_label = 'points',
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
    
def Constplot(year):
    
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
    dfC_S3['LapsLed'] = pd.DataFrame(dflaps[dflaps.position == 1].groupby('Team')['position'].sum())
    dfC_S3['LapsLed'] = dfC_S3['LapsLed'].fillna(0)
    dfC_S3.iloc[::,-4:] = dfC_S3.iloc[::,-4:].astype(int)

    
   
    df = dfC_S3
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
    
    
    
    
    
    dfw = fw(dfC_S3, 'wins')
    #sources
    sw = ColumnDataSource(dfw)
    
    #figure instance
    pw = figure(plot_height=160, plot_width=195,toolbar_location='below', tools = 'pan, wheel_zoom, box_zoom, reset',
                 x_range=(-1.2, 1.2), y_range=(.6,.1), x_axis_type=None, y_axis_type=None)

    #2021
    pw_ = pw.wedge(x=0, y=.5, radius=1, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='white', fill_color='c', line_width=.3,  source=sw)#, legend_field='Team'
    pw.add_tools(HoverTool(renderers=[pw_], tooltips=[('Team','@Team'), ('Wins','@wins'), ('','@pct{0.0} %'), ('year',str(year))],point_policy='follow_mouse'))
    
    #white
    pw.wedge(x=0, y=.505, radius=.6, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='#010727', line_width=.1, fill_color='#010727', source=sw)
    
    lw = Label(x=0, y=.49, x_offset=0, y_offset=0, text='Wins',
           text_color='white', text_font_size='15px', text_font_style='bold', text_align='center')
    pw.add_layout(lw)

    pw.outline_line_color=None
    pw.toolbar.autohide = True
    pw.background_fill_color = '#010727'
    pw.background_fill_alpha =.8
    pw.border_fill_color =  '#010727'
    pw.border_fill_alpha = .8
    
    
    dfp = fw(dfC_S3, 'poles')
    #sources
    sp = ColumnDataSource(dfp)
    
    #figure instance
    pp = figure(plot_height=160, plot_width=195,toolbar_location='below', tools = 'pan, wheel_zoom, box_zoom, reset',
                 x_range=(-1.2, 1.2), y_range=(.6,.1), x_axis_type=None, y_axis_type=None)

    #2021
    pp_ = pp.wedge(x=0, y=.5, radius=1, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='white', fill_color='c', line_width=.3,  source=sp)#, legend_field='Team'
    pp.add_tools(HoverTool(renderers=[pp_], tooltips=[('Team','@Team'), ('Poles','@poles'), ('','@pct{0.0} %'), ('year',str(year))],point_policy='follow_mouse'))

    
    #white
    pp.wedge(x=0, y=.505, radius=.6, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='#010727', line_width=.1, fill_color='#010727', source=sp)
    
    lp = Label(x=0, y=.49, x_offset=0, y_offset=0, text='Poles',
           text_color='white', text_font_size='15px', text_font_style='bold', text_align='center')
    pp.add_layout(lp)

    pp.outline_line_color=None
    pp.toolbar.autohide = True
    pp.background_fill_color = '#010727'
    pp.background_fill_alpha =.8
    pp.border_fill_color =  '#010727'
    pp.border_fill_alpha = .8
    
    
    dff = fw(dfC_S3, 'FastLap')
    #sources
    sf = ColumnDataSource(dff)
    
    pf = figure(plot_height=160, plot_width=195,toolbar_location='below', tools = 'pan, wheel_zoom, box_zoom, reset',
                 x_range=(-1.2, 1.2), y_range=(.6,.1), x_axis_type=None, y_axis_type=None)

    #2021
    pf_ = pf.wedge(x=0, y=.5, radius=1, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='white', fill_color='c', line_width=.3,  source=sf)#, legend_field='Team'
    pf.add_tools(HoverTool(renderers=[pf_], tooltips=[('Team','@Team'), ('Fast laps','@FastLap'), ('','@pct{0.0} %'), ('year',str(year))],point_policy='follow_mouse'))

    
    #white
    pf.wedge(x=0, y=.505, radius=.6, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='#010727', line_width=.1, fill_color='#010727', source=sf)
    
    lf = Label(x=0, y=.49, x_offset=0, y_offset=0, text='Fast laps',
           text_color='white', text_font_size='13px', text_font_style='bold', text_align='center')
    pf.add_layout(lf)

    pf.outline_line_color=None
    pf.toolbar.autohide = True
    pf.background_fill_color = '#010727'
    pf.background_fill_alpha =.8
    pf.border_fill_color =  '#010727'
    pf.border_fill_alpha = .8
    
    
    dfl = fw(dfC_S3, 'LapsLed')
    if year == 2020:
        dfl.iloc[0,-1] = dfl.iloc[0][-1] + dfl.loc['Williams'][-1]
        dfl.iloc[0,-2] = dfl.iloc[0][-2] + dfl.loc['Williams'][-2]
        dfl.iloc[0,-3] = dfl.iloc[0][-3] + dfl.loc['Williams'][-3]
        dfl = dfl[dfl.index != 'Williams']
    #sources
    sl = ColumnDataSource(dfl)
    
    #figure instance
    pl = figure(plot_height=160, plot_width=195,toolbar_location='below', tools = 'pan, wheel_zoom, box_zoom, reset',
                 x_range=(-1.2, 1.2), y_range=(.6,.1), x_axis_type=None, y_axis_type=None)

    #2021
    pl_ = pl.wedge(x=0, y=.5, radius=1, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='white', fill_color='c', line_width=.3,  source=sl)#, legend_field='Team'
    pl.add_tools(HoverTool(renderers=[pl_], tooltips=[('Team','@Team'), ('Laps led','@LapsLed'), ('','@pct{0.0} %'), ('year',str(year))],point_policy='follow_mouse'))

    
    #white
    pl.wedge(x=0, y=.505, radius=.6, start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
             line_color='#010727', line_width=.1, fill_color='#010727', source=sl)
    
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
    r = gridplot([[pw, pp, pf, pl]], merge_tools=True, toolbar_location='below', width=190, height=140)


    
    pL = LPlot(dfL, year)
    
    t1 = Panel(child=pCons, title='Overall')
    t2 = Panel(child =pL, title='By Grand Prix')
    tabs = Tabs(tabs= [t1, t2])
    
    col = column([tabs, r], align='start', spacing=-3)
    
    t = Panel(child=col, title=str(year))
    
    
    return t


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

