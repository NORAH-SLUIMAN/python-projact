import pandas as pd
import streamlit as st


df = pd.read_csv('store_sales.csv')


df['Total Sales'] = df['Price'] * df['Quantity_Sold']

st.title("Store Sales Dashboard")
st.write("This dashboard analyzes store sales data.")


st.subheader("Sales Data")
st.write(df)


total_sales_per_store = df.groupby('Store')['Total Sales'].sum().reset_index()


st.subheader("Total Sales by Store")
st.bar_chart(total_sales_per_store.set_index('Store'))

total_sales = df['Total Sales'].sum()
st.write(f"Total Sales: ${total_sales}")
