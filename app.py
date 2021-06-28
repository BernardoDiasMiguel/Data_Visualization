## Imports
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import datetime as dt

from pathlib import Path

# Input Data
input_folder = Path("./input/")
player_data = pd.read_csv(input_folder / 'player_data.csv').rename(columns={"name": "Player"})
seasons = pd.read_csv(input_folder / 'Seasons_Stats.csv').drop("Unnamed: 0", axis=1)

# Preparing datasets for each plot
# df - plot 1
plot1_data = player_data.groupby("college").count()["Player"].reset_index().rename(columns={"Player":"count"})

# df - plot 2
mvps = pd.read_csv(input_folder/"mvps.csv")
mvps["MVP"] = 1
bdps = pd.read_csv(input_folder/"bdps.csv")
bdps["BDP"] = 1
df2 = pd.merge(seasons, player_data, how="left", on="Player")[["Year", "Player", "college"]]
df2 = pd.merge(df2, mvps, how="left", on=["Player", "Year"])
df2 = pd.merge(df2, bdps, how="left", on=["Player", "Year"])
df2[["MVP", "BDP"]] = df2[["MVP", "BDP"]].fillna(0)

# df - plot 3
df_nba = pd.merge(seasons, player_data, on="Player", how="inner").fillna(0)
df_nba["PRA"] = (df_nba["PTS"] + df_nba["TRB"] + df_nba["AST"]).astype(int)
df_nba["Year"] = df_nba["Year"].astype(int)

# df - plot 4
df4 = pd.merge(seasons, player_data, on="Player", how="left")
df4["height"] = df4["height"].str.replace("-", ".").astype(float) * 0.3048
df4["weight"] = df4["weight"] * 0.4536
df4 = df4[(df4["height"].notna() & df4["weight"].notna() & df4["college"].notna())].groupby("college").mean().reset_index()[["weight", "height", "college"]]


## Interactive Component
all_colleges = [dict(label=college, value=college) for college in player_data['college'].unique()]


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

home_page = html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H4('Data Visualization - 2020/2021', className='row'),
                    html.Br(),
                    html.P("Analysis of what is the best college in basketball terms"),
                    dcc.Markdown("Dataset: [Source](https://www.kaggle.com/drgilermo/nba-players-stats)"),
                    html.Br(),
                    dcc.Markdown(
                    '''
                    #### Github
                    Find the code that serves this app at [ALTERAR LINK Github](https://github.com/xxxx/xxxxx).
                    '''),
                    html.Br(),
                    dcc.Markdown('''
                    #### Author
                    - **Bernardo Ferreira**, M20190450''')
                    ],className="row",style={'width': '98%', 'display': 'inline-block'})
                ]),

            html.Div([
                html.H4('Data Dictionary', className='row'),
                dcc.Markdown('''
                |     Variable     |    Type  |  Description                                                |
                |-------------------------|:-----------------:|-----------------------------------------------------------------------|
                | MVP | Numeric (int) | Most Valuable Player |
                | Col1 | Col2 | Col3 |
                '''),
                ],
                className="row", style={'width': '80%', 'display': 'inline-block'}),
        ],className="row"),
        ])

app.layout = html.Div([
    html.H1('NBA Stats', className='Title'),
    dcc.Tabs([
            dcc.Tab(label='Home', children=[
                home_page
            ]),
            dcc.Tab(label='Dashboard', children=[
                html.Div([
    html.Div([
        html.Div([
            html.H4('College Choice', className='h4'),
            dcc.Dropdown(
                id='college_drop',
                options=all_colleges,
                value=[],
                multi=True
            ),
            html.Br(),
            html.H4('Linear Log', className = 'h4'),
            html.P('Selecting log transforms continous indicators variables to better measure'),
            dcc.RadioItems(
                id='lin_log',
                options=[dict(label='Linear', value=0), dict(label='log', value=1)],
                value=0
            ),
        ], className='column1 pretty'),
        html.Div([
            html.Div([dcc.Graph(id='draftsbycollege')], className='bar_plot pretty'),
        ], className='column2')
    ], className='row'),
    html.Div([
        html.Div([dcc.Graph(id='mvpsbdps')], className='column3 pretty'),
        html.Div([dcc.Graph(id='heighweight')], className='column3 pretty')
    ], className='row'),
    html.Div([
        html.Div([dcc.Graph(id='prabyyear')], className='column3 pretty')
    ], className='row')

            ])
        ]),
    ])])

## Callbacks
@app.callback(
    [
         Output("draftsbycollege", "figure"),   
         Output("mvpsbdps", "figure"),
         Output("prabyyear", "figure"),
         Output("heighweight", "figure"),
    ],
    [
        Input("college_drop", "value"),
        Input("lin_log", "value"),
    ])


def plots(colleges, scale):
    
    ## First Plot - nº drafts por college
    plot1 = px.bar(
        plot1_data.where(plot1_data["college"].isin(colleges)), x="college", y="count",
        title="Players drafted by college")
                             
    ## Second Plot
    x = df2["college"].where(df2["college"].isin(colleges))
    y_mvp = df2["MVP"].where(df2["college"].isin(colleges))
    y_bdp = df2["BDP"].where(df2["college"].isin(colleges))
    plot2 = go.Figure()
    plot2.add_trace(go.Histogram(histfunc="sum", y=y_mvp, x=x, name="MVP"))
    plot2.add_trace(go.Histogram(histfunc="sum", y=y_bdp, x=x, name="BDP"))
    plot2.update_layout(title="MVPs and BDPs by College", xaxis_title="College", yaxis_title="Absolute Frequency")

    # Third Plot (Line Chart) - nº PRA ao longo dos anos por college
    plot3_data = df_nba.groupby(["Year", "college"]).sum()["PRA"].reset_index()
    plot3_data = plot3_data.loc[plot3_data.where(plot3_data["college"].isin(colleges)).notna().sum(axis=1) >= 3,:]
    # try/catch to check if 'colleges' is not filled
    try:
        plot3 = px.line(
            plot3_data, x="Year", y="PRA", color='college', log_y=bool(scale),
            title=["PRA by Year", "PRA by Year (log scaled)"][scale]
            )
    except KeyError:
        plot3 = go.Figure()
        plot3.update_layout(title=["PRA by Year", "PRA by Year (log scaled)"][scale])

    # Fourth Plot (Scatter Chart)
    plot4 = px.scatter(df4.loc[df4["college"].isin(colleges), :], x="height", y="weight", color="college",
        title="Average Height (m) and Weight (Kg) by College")

    return plot1, plot2, plot3, plot4


server = app.server

if __name__ == '__main__':
    app.run_server(debug=False)