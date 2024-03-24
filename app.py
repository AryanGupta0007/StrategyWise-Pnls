import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import time
import seaborn as sns
from main import client_1_PnL
from main import client_2_PnL
from main import orderTags
from main import df
from main import clients
dataframe = df
print(dataframe)
# print(df.columns)
print('yaha se')
# print(df.loc[df['LoginID'] == 'KB188'])
orderTags_Pnl = []
for orderTag in orderTags:
    current_df = df.loc[df['orderTag'] == orderTag]
    # print(current_df)
    amount = current_df['amount'].sum()
    orderTags_Pnl.append(amount)
    print(amount)
# print(df)
# import seaborn as sns
# import matplotlib.pyplot as plt

# create some example data
data = {'Strategy': orderTags,
        'PnL': orderTags_Pnl}

# convert data to a pandas DataFrame
df = pd.DataFrame(data).sort_values('PnL', ascending=False)
good_df = df.reset_index()
print(df)

# # pivot the DataFrame to create a matrix of PnL values by strategy
pnl_matrix = good_df.pivot(index=None, columns='Strategy', values='PnL')

# create a seaborn heatmap
sns.set()
sns.heatmap(pnl_matrix, cmap="YlGnBu", annot=True, fmt=".0f")

# customize the plot
plt.title("Strategy PnL Heatmap")
plt.xlabel("Strategy")
plt.ylabel("PnL")

# display the plot in Streamlit
st.pyplot()#
st.set_option('deprecation.showPyplotGlobalUse', False)
# fig = px.imshow(df)
#
# # customize the plot
# fig.update_layout(title="Strategy PnL Heatmap",
#                   xaxis_title="Strategy",
#                   yaxis_title="PnL")
#
# # display the plot in Streamlit
# st.plotly_chart(fig)

# def plot_heatmap():
# pnl_matrix = df.pivot_table(index="orderTag", columns="date", values="amount")
#
# # Create heatmap using seaborn
# st.title("Strategy-wise PnL heatmap")
# sns_plot = sns.heatmap(pnl_matrix)
# st.pyplot(sns_plot.figure)
def plot_bar_chart(title,df, label_angle=99, change_axis=False):
    st.title(title)
    # print(df)
    # Create a bar chart using Altair
    column_list = list(df.columns)
    if change_axis == True:
        chart = alt.Chart(df).mark_bar().encode(
            x=column_list[1],
            y=column_list[0]
        ).properties(width=400, height=800)
    else:
        chart = alt.Chart(df).mark_bar().encode(
            x=column_list[0],
            y=column_list[1]
        ).properties(width=400, height=300)

    if (label_angle != 99):
        chart = chart.configure_axis(labelAngle=label_angle)


    st.altair_chart(chart, use_container_width=True)

def plot_heatmap(df):
    st.title('Strategy Wise')
    sns_plot = sns.heatmap(df)

    st.pyplot(sns_plot.figure)
def plot_point_chart(title, point_labels, x_points, y_points, x_label, y_label):
    st.title(title)

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot the two points with different colors and labels
    colors = ['Black', 'Green', 'Blue', 'White', 'Grey', 'Yellow', 'Black', 'Green', 'Blue', 'White', 'Grey', 'Yellow',
              'Grey', 'Yellow']

    for x in range(len(x_points)):
        ax.scatter(x_points[x], y_points[x], color=colors[x], label=point_labels[x])
    # ax.scatter(x2, y2, color="blue", label="Client 2")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Add a legend to the plot
    ax.legend()

    # Display the plot in the Streamlit app
    st.pyplot(fig)
def create_table(df):
    # print(df)
    # print(df['PnL'])
    df['PnL'] = round(df['PnL'], 2)
    # print(df['PnL'])

    table_style = [{'selector': 'th',
                    'props': [('text-align', 'right')]},
                   {'selector': 'td',
                    'props': [('text-align', 'right')]}]

    st.table(df.style.set_table_styles(table_style))


# print(orderTags)
# Create a sample dataframe
options = ['Select Option','Broker',  'Order Tag Wise', 'Client & Strategy Wise']
charts = ['Select Chart' ,'Bar Chart', 'Line Chart']
selection = st.selectbox("Select an option:", options)

match selection:
    case 'Broker':
        st.title('Broker PnL')
        broker_pnl = client_1_PnL + client_2_PnL
        st.write('The total Pnl of the broker is ', broker_pnl)
        client_pnLs = []
        dates = []
        for client in clients:
            df = dataframe
            client_df = df.loc[df['LoginID'] == str(client)].reset_index()
            print(client_df)
        # time.sleep(10)
        # if 'date' in client_df.columns:
        #     # print('hi')
            datee = client_df.loc[0, 'date']
            print(client_df['date'])

            client_pnl = client_df['amount'].sum()
            client_pnLs.append(client_pnl)
            dates.append(datee)

        df = pd.DataFrame({
            'Clients': clients,
            'PnL': client_pnLs
        })
        plot_bar_chart(title=selection, df=df, label_angle=0)

    case 'Order Tag Wise':
        strategy_pnLs = []
        dates = []
        for orderTag in orderTags:
            df = dataframe
            strategy_df = df.loc[df['orderTag'] == orderTag]
            print(strategy_df)
            strategy_pnl = strategy_df['amount'].sum()
            strategy_pnLs.append(strategy_pnl)
            datee = df.loc[0, 'date']
            dates.append(datee)
        df = pd.DataFrame({
            'Strategies': orderTags,
            'PnL': strategy_pnLs
        })
        # plot_heatmap(df=df)
        plot_bar_chart(title=selection, df=df, change_axis=True)
            # case 'Point Chart':
            #     plot_point_chart(title=selection, x_points=strategy_pnLs, y_points=dates,
            #                      x_label='PnL', y_label='Date',
            #                      point_labels=orderTags)
        Dict = {
            'Strategies': 'Total PnL',
            'PnL': sum(strategy_pnLs)
        }
        df2 = pd.DataFrame(Dict, columns=['Strategies', 'PnL'], index=[0])
        df.sort_values('PnL', ascending=False, inplace=True)
        # print(df)
        df.reset_index(inplace=True)
        df.drop(['index'], inplace=True, axis=1)
        df = pd.concat([df, df2], ignore_index=True)
        create_table(df)
    case 'Client & Strategy Wise':

        select_a_client = ['Select a Client ']

        for client in clients:
            select_a_client.append(client)

        select_client = st.selectbox("Select a Client:", select_a_client)
        df = dataframe
        client_data = df.loc[df['LoginID'] == select_client]
        strategy_pnLs = []
        dates = []
        for orderTag in orderTags:
            strategy_df = client_data.loc[df['orderTag'] == orderTag]
            # print(strategy_df)
            strategy_pnl = strategy_df['amount'].sum()
            strategy_pnLs.append(strategy_pnl)
            datee = df.loc[0, 'date']
            dates.append(datee)
        df = pd.DataFrame({
            'Strategies': orderTags,
            'PnL': strategy_pnLs
        })
        df['PnL'] = round(df['PnL'], 2)
        plot_bar_chart(title=selection, df=df, change_axis=True)
        Dict = {
            'Strategies': 'Total PnL',
            'PnL': sum(strategy_pnLs)
        }
        df2 = pd.DataFrame(Dict, columns=['Strategies', 'PnL'], index=[0])
        df.sort_values('PnL', ascending=False, inplace=True)
        # print(df)
        df.reset_index(inplace=True)
        df.drop(['index'], inplace=True, axis=1)
        df = pd.concat([df, df2], ignore_index=True)
        create_table(df)
# strategy_selection = st.selectbox('Select a strategy ', orderTags)