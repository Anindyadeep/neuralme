############################## THE APP PORTION TO PLOT THE THINGS OF THE DATA ##############################

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from custom_data_img import *


def visualise(choice):
    if choice == 'Visualisations':
        st.subheader("The Plot of the data and visualisation")
        given_options = ['Titanic', 'Breast Cancer']
        option = st.selectbox("Please select the .csv datasets: ", given_options)

        if option == 'Titanic':
            st.write("WE ARE SHOWING THE STATS OF THE TRINING DATA")
            
            path_train = r'Datasets/Titanic/train.csv'
            train = pd.read_csv(path_train)
            st.dataframe(train.head())

            st.markdown("### The pie plot of the stats of survival of the passengers")
            fig = px.pie(train, names='Survived', title='Passenger Survival')
            st.plotly_chart(fig)

            st.markdown("### The pie plots of the stats three different contexts")
            st.markdown("#### 1. The survival ratio in the category of embarked C category")
            st.markdown("#### 2. The survival ratio in the category of embarked S category")
            st.markdown("#### 3. The survival ratio in the category of embarked Q category")

            fig = make_subplots(rows=1, cols=3, specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]])
            fig.add_trace(
                go.Pie(labels=train.loc[train['Embarked'] == 'C']['Survived'], pull=[.1, .1],
                       title='Embarked C vs. Survived'), row=1, col=1)

            fig.add_trace(
                go.Pie(labels=train.loc[train['Embarked'] == 'S']['Survived'], pull=[.07, .07],
                       title='Embarked S vs. Survived'), row=1, col=2)

            fig.add_trace(
                go.Pie(labels=train.loc[train['Embarked'] == 'Q']['Survived'], pull=[.1, .1],
                       title='Embarked Q vs. Survived'), row=1, col=3)

            fig.update_layout(height=500, width=800)
            st.plotly_chart(fig)

            st.markdown("### The histogram plots of the stats of the probability density of the age category")
            fig = px.histogram(train, x='Age', nbins=50, histnorm='probability density')
            st.plotly_chart(fig)

        elif option == 'Breast Cancer':
            path = r'Datasets/Breast cancer/data.csv'
            train = pd.read_csv(path)
            st.write("WE ARE SHOWING THE STATS OF THE TRINING DATA OF BREAST CANCER")
            st.dataframe(train.head())
            st.markdown("### The plot of the the Bengin vs Malignant density in a histogram")
            fig = px.histogram(train, x='diagnosis', nbins=50, histnorm='probability density')
            st.plotly_chart(fig)
