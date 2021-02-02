############################## THE ABOUT THE PROJECT SECTION ##############################

import streamlit as st


def about_project():
    st.markdown("## WELCOME EVERYONE TO NEURAL.ME")
    st.write("Here I will show you some demo of the training and the testing of the simple ANN model that I "
             "have made from scratch, by only using numpy, no tensorflow, no keras etc")
    st.markdown("#### This is the version 1.0.0, just the new born baby, this version can do the following things:")
    st.markdown("1. You can see some data, 2 toys, 2 some what real life and one real life data for binary "
                "classification")
    st.markdown("2. You can visualise the data in the next section for some simple insights to gain")
    st.markdown("3. The most interesting part, where you can train the model and see the model to train in some "
                "epochs that is already set up along with the model architecture, but all the neural network "
                "archirecture alog with all other required model info are provided there. Also you cab have a look to "
                "a really cool animation of how the loss is decreasing both train and validation and how the accuracy "
                "is increasing")
    st.markdown("### Some cons of the project (temporary)")
    st.markdown("1. You can not upload a custom data to train the model as that part is not being developed in "
                "the backend for now.")
    st.markdown("2. You can not set the model configuration yourself like the number of hidden layers, "
                "hidden neurons in each layer etc")
    st.markdown("3.The model to become succesful in training is kinda chance factor of 80 % of success rate, "
                "i.e. when ever the model  is giving nan that means the random weights initialised in the backend is "
                "not at all fruitful, so please clear the cache and rerun the model, also you can not do multiple "
                "classification in this version")

    st.markdown("### I hope you like the project and I will try to cover up the shortcomings now in the next version "
                "and later and so on...")
