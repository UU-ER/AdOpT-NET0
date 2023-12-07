import pandas as pd
from dash import Dash, dcc, html
from dash.dash_table import FormatTemplate, DataTable

def create_dashboard(results):

    technology_sizes = results.technologies

    app = Dash(__name__)

    app.layout = html.Div(
        children=[
            html.H1(children="Technologies"),
            # html.P(
            #     children=(
            #         "Analyze the behavior of avocado prices and the number"
            #         " of avocados sold in the US between 2015 and 2018"
            #     ),
            # ),
            DataTable(technology_sizes.to_dict('records'))
            # dcc.Graph(
            #     figure={
            #         "data": [
            #             {
            #                 "x": data["Date"],
            #                 "y": technology_sizes,
            #                 "type": "lines",
            #             },
            #         ],
            #         "layout": {"title": "Average Price of Avocados"},
            #     },
            # ),
            # dcc.Graph(
            #     figure={
            #         "data": [
            #             {
            #                 "x": data["Date"],
            #                 "y": data["Total Volume"],
            #                 "type": "lines",
            #             },
            #         ],
            #         "layout": {"title": "Avocados Sold"},
            #     },
            # ),
        ]
    )

    # if __name__ == "__main__":
    app.run_server(debug=True)