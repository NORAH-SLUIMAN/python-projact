import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

# بيانات عينة للمنتجات
data = {
    'product': ['A5JLAU2ARJ0BO', 'ADLVFFE4VBT8', 'A3OXHLG6DIBRW8', 'A6FIAB28IS79', 'A680RUE1FDO8B'],
    'count': [520, 501, 498, 431, 406],
}
df = pd.DataFrame(data)

app = dash.Dash(__name__)


orange_color = '#FF9900' 

app.layout = html.Div(style={'backgroundColor': '#f8f8f8', 'padding': '20px'}, children=[
   
    dcc.Input(id='search-input', type='text', placeholder='بحث عن منتج...', style={'width': '300px', 'padding': '10px'}),
    html.Div(id='search-output'),

    html.Img(src='https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg',
              style={'width': '200px', 'display': 'block', 'margin': '0 auto'}),  # زيادة حجم الشعار

    html.Div(children=[
        dcc.Graph(
            id='product-graph',
            figure=px.bar(df, x='product', y='count', title='عدد المبيعات حسب المنتج',
                          labels={'product': 'المنتج', 'count': 'عدد المبيعات'}, 
                          color_discrete_sequence=[orange_color])  # استخدام اللون البرتقالي
        )
    ], style={'margin-top': '20px'}),
])

@app.callback(
    Output('search-output', 'children'),
    Input('search-input', 'value')
)
def update_output(value):
    return f'نتائج البحث عن: {value}' if value else ''

if __name__ == '__main__':
    app.run_server(debug=True)
