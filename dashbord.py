import pandas as pd
from dash import dcc, html, Dash, dash_table
import plotly.express as px

#  بيانات حجوزات الفنادق
df = pd.read_csv('/Users/nona/Desktop/hotel_bookings.csv')

# حساب عدد الزيارات لكل فندق
hotel_visits = df.groupby('hotel')['hotel'].count().reset_index(name='عدد الزيارات')

bar_fig = px.bar(hotel_visits, x='hotel', y='عدد الزيارات', 
                  title='أعلى الفنادق زيارة', 
                  color='عدد الزيارات', 
                  color_continuous_scale=px.colors.sequential.Blues,
                  labels={'عدد الزيارات': 'عدد الزيارات', 'hotel': 'اسم الفندق'})


monthly_visits = df.groupby(['hotel', 'arrival_date_month']).size().reset_index(name='عدد الزيارات')
monthly_visits = monthly_visits[monthly_visits['arrival_date_month'] == 'October'] 

pie_fig = px.pie(monthly_visits, names='hotel', values='عدد الزيارات', 
                  title='توزيع عدد الزيارات حسب الفندق',
                  color_discrete_sequence=px.colors.sequential.RdBu)


app = Dash(__name__)
app.layout = html.Div(style={'backgroundColor': '#f5f5f5', 'padding': '20px'}, children=[
    html.H1(children=' حجوزات الفنادق', style={'textAlign': 'center', 'color': '#1f77b4'}),
    

    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}, children=[
        dcc.Graph(
            id='top-hotels-visited',
            figure=bar_fig,
            style={'flex': '1', 'marginRight': '20px', 'width': '400px'}  # تحديد عرض المخطط
        ),

        dcc.Graph(
            id='hotel-visit-distribution',
            figure=pie_fig,
            style={'flex': '1', 'marginLeft': '20px', 'width': '400px'}  # تحديد عرض المخطط
        )
    ]),

    dash_table.DataTable(
        id='monthly-visits-table',
        columns=[{"name": i, "id": i} for i in monthly_visits.columns],
        data=monthly_visits.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'backgroundColor': '#f9f9f9',
            'color': '#333'
        },
        style_header={
            'backgroundColor': '#1f77b4',
            'color': 'white'
        }
    ),

    html.Div(children='''معلومات عامة عن البيانات:
    يتضمن هذا التحليل علئ تحليل بيانات حجوزات الفنادق، ويظهر عدد الزيارات لكل فندق.
    ''', style={'textAlign': 'center', 'color': '#333'})
])

if __name__ == '__main__':
    app.run_server(debug=True)
