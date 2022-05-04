from plots1 import tabs

import json

import os
from flask import Flask, render_template, request
from tornado.ioloop import IOLoop
import os
from jinja2 import Environment, FileSystemLoader
#from flask.ext.widgets import Widgets
#from werkzeug import secure_filename
#from sqlalchemy import SQLAlchemy
from bokeh.embed import components, server_document
from bokeh.resources import CDN
import bokeh
from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show, save
from bokeh.models import ColumnDataSource, Slider, Span, CustomJS, RangeSlider, TextInput, Button
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.models.widgets import Tabs, Panel, Paragraph, Div
from bokeh.themes import built_in_themes
from bokeh.io import curdoc
from bokeh.models.tools import CrosshairTool, HoverTool




app = Flask(__name__)



@app.route('/')
def index1():
    

    script, div = components(tabs())
    return render_template('index.html', script=script, div=div, resources=CDN.render())
      


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
