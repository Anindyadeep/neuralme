############################## THE APP DATA FETCHING AREA ##############################

import matplotlib
from visualise_custom_data import *
from sklearn.datasets import make_moons, make_circles
import os

matplotlib.use('Agg')


def fetch_data(choice):
    if choice == 'Data':

        st.subheader("The Data")

        st.write('Currently I have tested on four datasets, two belong to some what real world , and two are toy '
                 'dataset, If you want to use any other custom CSV dataset, then it should have only one file. No '
                 'separate csv files are to be taken in this version')

        given_options = ['Titanic', 'Breast Cancer', 'Moons', 'Circles', 'Custom Data']
        option = st.selectbox("Please select the .csv datasets: ", given_options)
        st.write('You selected: ', option)

        if option != 'Custom Data':
            if option == 'Titanic':
                path_train = r'Datasets/Titanic/train.csv'
                path_test = r'Datasets/Titanic/test.csv'
                path_val = r'Datasets/Titanic/gender_submission.csv'
                train_data = pd.read_csv(path_train)
                test_data = pd.read_csv(path_test)
                test_val = pd.read_csv(path_val)
                test_data = pd.concat([test_data, test_val], axis=1)

                st.subheader("TRAINING DATA")

                if st.checkbox("Show training data"):
                    st.write("THE TRAINING DATA")
                    st.dataframe(train_data.head())
                if st.checkbox("Show shape of train data"):
                    st.write(train_data.shape)
                if st.checkbox("Show columns of the train data"):
                    all_columns = train_data.columns.to_list()
                    st.write(all_columns)
                if st.checkbox("Show summary of the train data"):
                    st.write(train_data.describe())
                if st.checkbox("Show value counts of the train data"):
                    st.write(train_data.iloc[:, 1].value_counts())

                st.subheader("TESTING DATA")
                if st.checkbox("Show testing data"):
                    st.write("THE TESTING DATA")
                    st.dataframe(test_data.head())
                if st.checkbox("Show shape of the test data"):
                    st.write(test_data.shape)
                if st.checkbox("Show columns of the test data"):
                    all_columns = test_data.columns.to_list()
                    st.write(all_columns)
                if st.checkbox("Show summary of the test data"):
                    st.write(test_data.describe())
                if st.checkbox("Show value counts of the test data"):
                    st.write(test_data.iloc[:, 12].value_counts())

            elif option == 'Breast Cancer':
                path = r'Datasets/Breast cancer/data.csv'
                data = pd.read_csv(path)
                st.write("Here there is no train or test dataframes but we can split those with some of the custom "
                         "made functions further")

                st.subheader("GETTING THE DATA")
                if st.checkbox("Show data"):
                    st.write("THE DATA")
                    st.dataframe(data.head())
                if st.checkbox("Show shape of the data"):
                    st.write(data.shape)
                if st.checkbox("Show columns of the data"):
                    all_columns = data.columns.to_list()
                    st.write(all_columns)
                if st.checkbox("Show summary of the data"):
                    st.write(data.describe())
                if st.checkbox("Show value counts of the data"):
                    st.write(data.iloc[:, 1].value_counts())

            elif option == 'Moons':
                X, Y = make_moons(n_samples=10000, noise=0.2)
                data = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=Y))
                st.write("It is totally obvious that, the dataframe really makes not that much sense since this is a "
                         "toy data, so I will also try to show the plots along side in order to make some more "
                         "analysis")
                if st.checkbox("Show the data"):
                    st.dataframe(data.head())

                if st.checkbox("Visualise the data"):
                    plot_toy_data('moons', samples=1000, noise=0.3, stream=True)

            elif option == 'Circles':
                X, Y = make_circles(n_samples=1000, noise=0.1)
                data = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=Y))
                st.write("It is totally obvious that, the dataframe really makes not that much sense since this is a "
                         "toy data, so I will also try to show the plots along side in order to make some more "
                         "analysis")
                if st.checkbox("Show the data"):
                    st.dataframe(data.head())

                if st.checkbox("Visualise the data"):
                    plot_toy_data('circles', samples=10000, noise=0.07, stream=True)

        else:
            uploaded_files = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=True)
            if uploaded_files:
                for file in uploaded_files:
                    file.seek(0)
                uploaded_data_read = [pd.read_csv(file) for file in uploaded_files]
                raw_data = pd.concat(uploaded_data_read)
                st.dataframe(raw_data.head())

            st.write("This portion is under development, so kindly use that later in the future updates")
