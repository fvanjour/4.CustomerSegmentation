# Import libraries
import dash
from dash import dcc
from dash import html
import dash_table
import plotly.express as px
from sklearn.cluster import KMeans
import os
import datetime as dt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

# Read CSV and load to dataframe
os.chdir("c:\\Users\\Min Dator\\NodBootcamp\\BC#3\\Projects\\4. CustomerSegmentation\\data")
sales = pd.read_csv("online_retail_II.csv", low_memory=False)
df=sales.copy()

# Data Prep

# Populate Monetary
total_sales_per_customer = df.groupby('Customer ID').apply(lambda x: (x['Quantity'] * x['Price']).sum()).reset_index(name='Monetary')

# Populate Frequency
unique_invoices_per_customer = df.groupby('Customer ID')['Invoice'].nunique().reset_index(name='Frequency')

# Populate Recency
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format='%m/%d/%y %I:%M %p')
max_date = max(df["InvoiceDate"])
df["recency"] = max_date - df["InvoiceDate"]
recency_per_customer = df.groupby('Customer ID')['recency'].min().reset_index(name='Recency')
recency_per_customer["Recency"] = recency_per_customer["Recency"].dt.days

# Merge Customer Details (Recency, Frequency & Monetary)
customer_details = total_sales_per_customer.merge(unique_invoices_per_customer, on='Customer ID').merge(recency_per_customer, on='Customer ID')

# Remove Outliers using Interquartile Range (IQR)
rfm_columns = ['Monetary','Frequency','Recency']
for column in rfm_columns:
    Q1 = customer_details[column].quantile(0.25)
    Q3 = customer_details[column].quantile(0.75)
    IQR = Q3 - Q1

    # Defining bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtering outliers
    customer_details = customer_details[(customer_details[column] >= lower_bound) & (customer_details[column] <= upper_bound)]

# Create the clusters
pipe = make_pipeline(StandardScaler(), KMeans(n_clusters=4, random_state=42))
pipe.fit(customer_details[['Recency','Frequency','Monetary']])
customer_details["Cluster"] = pipe["kmeans"].labels_

cluster_map = {
    2: "High Value Active",
    0: "Engaged Mid-Tier",
    3: "Inactive Low Value",
    1: "New/Returning Low Value"
}

customer_details['Cluster'] = customer_details['Cluster'].map(cluster_map)

# Visualize Cluster with Pair Plots
fig1 = px.scatter_matrix(customer_details, dimensions=["Recency", "Frequency", "Monetary"], color="Cluster")
"""fig1 = px.scatter_matrix(customer_details, dimensions=["Recency", "Frequency", "Monetary"],
    color="Cluster",
    color_discrete_map={
        'High Value': 'blue',
        'Engaged Mid-Tier': 'green',
        'New/Returning Low Value': 'red',
        'Inactive Low Value': 'yellow'
    })"""

# Visualize Cluster with 3D Scatter Plot
fig2 = px.scatter_3d(customer_details, x='Recency', y='Frequency', z='Monetary', color='Cluster') 

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([

    # Title Slide
    html.Div(
        style={
            'background': '#FFF5E1',
            'borderRadius': '5px',
            'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.15)',
            'padding': '20px',
            'marginBottom': '30px'
        },
        children=[
            html.H1("Customer Segmentation", style={'textAlign': 'center'}),
            html.H3("Recency, Frequency and Monetary (RFM)", style={'textAlign': 'center'}),
            html.P("by: Franklin Vanjour", style={'textAlign': 'center', 'fontSize': '16px'})
        ]
    ),

     # Intro Slide
    html.Div(
        style={
            'background': '#FFF5E1',
            'borderRadius': '5px',
            'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.15)',
            'padding': '20px',
            'marginBottom': '30px'
        },
        children=[
            html.H2("Tom's Online Market Palce", style={'textAlign': 'center'}),
            html.Ul(children=[
                html.Li("Tom has an Online Market Place where he sells general merchandise"),
                html.Li("He wants to improve his business by gaining a better understanding of his customer segments"),
                html.Li("Tom requests Frank, a Data Analyst, to help with the Analysis")
            ]),
            html.Img(src='/assets/intro.png', width="254", height="198"),
            html.P("An extract of Tom's Sales Data:", style={'textAlign': 'center', 'fontSize': '16px'}),
            html.Div(
                children=[
                    dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in sales.columns],
                        data=sales.head().to_dict('records'),
                        )
                    ]
                )
        ]
    ),

    # Identified CLusters
    html.Div(
        style={
            'background': '#FFF5E1',
            'borderRadius': '5px',
            'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.15)',
            'padding': '20px',
            'marginBottom': '30px'
        },
        children=[
            html.H1("Customer Segments Found", style={'textAlign': 'center'}),
            html.Table([
            # Header
            html.Thead([
                html.Tr([html.Th("Segment"), html.Th("Segment Details"), html.Th("Recommended Action")])
                     ]),
        
             # Body
            html.Tbody([
                html.Tr([html.Td("High Value Active", style={'font-weight': 'bold'}), html.Td("High Monetary, High Frequency, High Recency"), html.Td("Reward with loyalty programs, and capitalize on cross-selling and up-selling opportunities.")]),
                html.Tr([html.Td("Engaged Mid-Tier", style={'font-weight': 'bold'}), html.Td("Medium Monetary, High Frequency, High Recency"), html.Td("Promote higher-tier offerings and enhance their brand connection with exclusive content or referral incentives.")]),
                html.Tr([html.Td("Inactive Low Value", style={'font-weight': 'bold'}), html.Td("Low Monetary, Low Recency, Low Frequency"), html.Td("Initiate reactivation campaigns, gather feedback, and introduce budget-friendly offerings.")]),
                html.Tr([html.Td("New/Lapsed Low Value", style={'font-weight': 'bold'}), html.Td("Low Monetary, High Recency, Low Frequency"), html.Td("Offer memorable onboarding experiences for newcomers and win-back campaigns for returning customers.")]),
        ])
    ]),
            html.Div(
                children=[
                    dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in customer_details.columns],
                        data=customer_details.head().to_dict('records'),
                        )
                    ]
                )
        ]
    ),

    # Pair Plots Slide
    html.Div(
        style={
            'background': '#FFFFFF',
            'borderRadius': '5px',
            'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.15)',
            'padding': '20px',
            'marginTop': '30px'
        },
        children=[
            html.H2("Pairwise RFM Relationships", style={'textAlign': 'center'}),
            html.Div(
                children=[
                    dcc.Graph(figure=fig1)
                    ]
                )
            ]
        ),

    # 3D Scatter Plot Slide
    html.Div(
        style={
            'background': '#FFFFFF',
            'borderRadius': '5px',
            'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.15)',
            'padding': '20px',
            'marginTop': '30px'
        },
        children=[
            html.H2("Customer Segments", style={'textAlign': 'center'}),
            html.Div(
                children=[
                    dcc.Graph(figure=fig2)
                    ]
                )
            ]
        ),


    # Highlights/Challenges Slide
    html.Div(
        style={
            'background': '#FFFFFF',
            'borderRadius': '5px',
            'boxShadow': '0px 0px 10px rgba(0, 0, 0, 0.15)',
            'padding': '20px',
            'marginTop': '30px'
        },
        children=[
            html.H2("Highlights / Challenges", style={'textAlign': 'center'}),
            html.Ul(children=[
                html.Li("Identifying a Good Dataset"),
                html.Li("Identifying which plots to Visualize"),
                html.Li("Ensuring a correct flow in the Story"),
                html.Li("Time Management"),                
                ])
            ]
        )
    ],
    style={
        'background': '#F2F3F4',
        'padding': '30px',
        'font-family': 'Arial'
        }
)

# Run the app
app.run_server(debug=False)
