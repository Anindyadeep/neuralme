############################## THE NEURAL NETWORK MODEL BUILD AND TRAINING IN THE APP ##############################

from custom_data_img import *
from custom_data_non_img import *
from backward import *
from forward import *
from architecture import *
import pandas as pd
import os


def model_build():
    given_data_options = ['Titanic', 'Breast Cancer', 'Moons', 'Circles']
    option = st.selectbox("Please select the .csv datasets: ", given_data_options)


        ###################################### BREAST CANCER DATA ######################################

    if option == 'Breast Cancer':
        path = r'Datasets/Breast cancer/data.csv'
        X_train, Y_train, X_val, Y_val, X_test, Y_test = get_breast_cancer(path)

        st.markdown("## The shapes of the numpy arrays are as follows")
        st.write("The shape of X_train is: ", X_train.shape)
        st.write("The shape of X_validation is: ", X_val.shape)
        st.write("The shape of X_test is: ", X_test.shape)
        st.write("The shape of Y_train is: ", Y_train.shape)
        st.write("The shape of Y_validation is: ", Y_val.shape)
        st.write("The shape of Y_test is: ", Y_test.shape)

        nodes = [X_train.shape[0], 20, 7, 5, 1]
        act_func = ['relu', 'relu', 'relu', 'sigmoid']
        learning_rate = 0.03
        epochs = 900

        st.write("### MODEL INFORMATION")
        st.write("No of hidden layers excluding the output layer: ", len(nodes) - 2)
        st.write("#### LAYERS INFORMATION")

        layers = ["1st", "2nd", "3rd", "4th"]
        neuron = [20, 7, 5, 1]
        func = ['relu', 'relu', 'relu', 'sigmoid']
        model_info = pd.DataFrame(list(zip(layers, neuron, func)),
                                  columns=["Layer no:", "Neuron in each layer", "actiation func"])
        st.dataframe(model_info)

        params = get_params(nodes, act_func)
        A_test, cache_test = full_feed_forward(X_test, params)
        st.write("CURRENT LOSS TEST: ", binary_loss(A_test, Y_test))
        st.write("CURRENT ACC TEST: ", accuracy(A_test, Y_test))

        st.markdown("## THE TRAINING")
        back_prop = {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_val": X_val,
            "Y_val": Y_val,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "params": params
        }
        params, LOSS_TRAIN, ACC_TRAIN, LOSS_VAL, ACC_VAL, EPOCH = train(back_prop, stream=True)
        metrics = pd.DataFrame(list(zip(EPOCH, LOSS_TRAIN, LOSS_VAL, ACC_TRAIN, ACC_VAL)),
                               columns=['EPOCHS', 'LOSS_TRAIN', 'LOSS_VAL', 'ACC_TRAIN', 'ACC_VAL'])
        st.write("### THE LOSS, AND THE ACCURACY PLOTS OF THE MODEL IN BOTH TRAINING AND VALIDATION SCENARIO")
        plot_loss_animation(metrics)
        plot_acc_animation(metrics)

        ###################################### TITANIC DATA ######################################

    elif option == 'Titanic':
        PATH1 = r'Datasets/Titanic/train.csv'
        PATH2 = r'Datasets/Titanic/test.csv'
        PATH_VAL = r'Datasets/Titanic/gender_submission.csv'

        X_train, Y_train, X_val, Y_val, X_test, Y_test = generate_titanic(PATH1, PATH2, PATH_VAL)

        st.markdown("## The shapes of the numpy arrays are as follows")
        st.write("The shape of X_train is: ", X_train.shape)
        st.write("The shape of X_validation is: ", X_val.shape)
        st.write("The shape of X_test is: ", X_test.shape)
        st.write("The shape of Y_train is: ", Y_train.shape)
        st.write("The shape of Y_validation is: ", Y_val.shape)
        st.write("The shape of Y_test is: ", Y_test.shape)

        nodes = [X_train.shape[0], 16, 32, 1]
        act_func = ['relu', 'relu', 'sigmoid']
        learning_rate = 0.03
        epochs = 500

        st.write("### MODEL INFORMATION")
        st.write("No of hidden layers excluding the output layer: ", len(nodes) - 2)
        st.write("#### LAYERS INFORMATION")

        layers = ["1st", "2nd", "3rd"]
        neuron = [16, 32, 1]
        func = ['relu', 'relu', 'sigmoid']
        model_info = pd.DataFrame(list(zip(layers, neuron, func)),
                                  columns=["Layer no:", "Neuron in each layer", "actiation func"])
        st.dataframe(model_info)

        params = get_params(nodes, act_func)
        A_test, cache_test = full_feed_forward(X_test, params)
        st.write("CURRENT LOSS TEST: ", binary_loss(A_test, Y_test))
        st.write("CURRENT ACC TEST: ", accuracy(A_test, Y_test))

        st.markdown("## THE TRAINING")
        back_prop = {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_val": X_val,
            "Y_val": Y_val,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "params": params
        }
        params, LOSS_TRAIN, ACC_TRAIN, LOSS_VAL, ACC_VAL, EPOCH = train(back_prop, stream=True)
        metrics = pd.DataFrame(list(zip(EPOCH, LOSS_TRAIN, LOSS_VAL, ACC_TRAIN, ACC_VAL)),
                               columns=['EPOCHS', 'LOSS_TRAIN', 'LOSS_VAL', 'ACC_TRAIN', 'ACC_VAL'])
        st.write("### THE LOSS, AND THE ACCURACY PLOTS OF THE MODEL IN BOTH TRAINING AND VALIDATION SCENARIO")
        plot_loss_animation(metrics)
        plot_acc_animation(metrics)

        ###################################### CIRCLES DATA ######################################

    elif option == 'Circles':
        X_train, Y_train, X_val, Y_val, X_test, Y_test = get_circle_data(10000, 0.07)

        st.markdown("## The shapes of the numpy arrays are as follows")
        st.write("The shape of X_train is: ", X_train.shape)
        st.write("The shape of X_validation is: ", X_val.shape)
        st.write("The shape of X_test is: ", X_test.shape)
        st.write("The shape of Y_train is: ", Y_train.shape)
        st.write("The shape of Y_validation is: ", Y_val.shape)
        st.write("The shape of Y_test is: ", Y_test.shape)

        nodes = [X_train.shape[0], 16, 16, 1]
        act_func = ['relu', 'relu', 'sigmoid']
        learning_rate = 0.075
        epochs = 500

        st.write("### MODEL INFORMATION")
        st.write("No of hidden layers excluding the output layer: ", len(nodes) - 2)
        st.write("#### LAYERS INFORMATION")

        layers = ["1st", "2nd", "3rd"]
        neuron = [16, 16, 1]
        func = ['relu', 'relu', 'sigmoid']
        model_info = pd.DataFrame(list(zip(layers, neuron, func)),
                                  columns=["Layer no:", "Neuron in each layer", "activation func"])
        st.dataframe(model_info)

        params = get_params(nodes, act_func)
        A_test, cache_test = full_feed_forward(X_test, params)
        st.write("CURRENT LOSS TEST: ", binary_loss(A_test, Y_test))
        st.write("CURRENT ACC TEST: ", accuracy(A_test, Y_test))

        st.markdown("## THE TRAINING")
        back_prop = {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_val": X_val,
            "Y_val": Y_val,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "params": params
        }
        params, LOSS_TRAIN, ACC_TRAIN, LOSS_VAL, ACC_VAL, EPOCH = train(back_prop, stream=True)
        metrics = pd.DataFrame(list(zip(EPOCH, LOSS_TRAIN, LOSS_VAL, ACC_TRAIN, ACC_VAL)),
                               columns=['EPOCHS', 'LOSS_TRAIN', 'LOSS_VAL', 'ACC_TRAIN', 'ACC_VAL'])
        st.write("### THE LOSS, AND THE ACCURACY PLOTS OF THE MODEL IN BOTH TRAINING AND VALIDATION SCENARIO")
        plot_loss_animation(metrics)
        plot_acc_animation(metrics)
        st.write("### THE DECISION BOUNDARY")
        st.pyplot(plot_decision_boundary(lambda x: predict(params, x.T), X_test, Y_test))

        ###################################### MOONS DATA ######################################

    elif option == 'Moons':
        X_train, Y_train, X_val, Y_val, X_test, Y_test = get_moons_data(10000, 0.2)

        st.markdown("## The shapes of the numpy arrays are as follows")
        st.write("The shape of X_train is: ", X_train.shape)
        st.write("The shape of X_validation is: ", X_val.shape)
        st.write("The shape of X_test is: ", X_test.shape)
        st.write("The shape of Y_train is: ", Y_train.shape)
        st.write("The shape of Y_validation is: ", Y_val.shape)
        st.write("The shape of Y_test is: ", Y_test.shape)

        nodes = [X_train.shape[0], 20, 7, 5, 1]
        act_func = ['relu', 'relu', 'relu', 'sigmoid']
        learning_rate = 0.075
        epochs = 1000

        st.write("### MODEL INFORMATION")
        st.write("No of hidden layers excluding the output layer: ", len(nodes) - 2)
        st.write("#### LAYERS INFORMATION")

        layers = ["1st", "2nd", "3rd", "4th"]
        neuron = [20, 7, 5, 1]
        func = ['relu', 'relu', 'relu', 'sigmoid']
        model_info = pd.DataFrame(list(zip(layers, neuron, func)),
                                  columns=["Layer no:", "Neuron in each layer", "activation func"])
        st.dataframe(model_info)

        params = get_params(nodes, act_func)
        A_test, cache_test = full_feed_forward(X_test, params)
        st.write("CURRENT LOSS TEST: ", binary_loss(A_test, Y_test))
        st.write("CURRENT ACC TEST: ", accuracy(A_test, Y_test))

        st.markdown("## THE TRAINING")
        back_prop = {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_val": X_val,
            "Y_val": Y_val,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "params": params
        }
        params, LOSS_TRAIN, ACC_TRAIN, LOSS_VAL, ACC_VAL, EPOCH = train(back_prop, stream=True)
        metrics = pd.DataFrame(list(zip(EPOCH, LOSS_TRAIN, LOSS_VAL, ACC_TRAIN, ACC_VAL)),
                               columns=['EPOCHS', 'LOSS_TRAIN', 'LOSS_VAL', 'ACC_TRAIN', 'ACC_VAL'])
        st.write("### THE LOSS, AND THE ACCURACY PLOTS OF THE MODEL IN BOTH TRAINING AND VALIDATION SCENARIO")
        plot_loss_animation(metrics)
        plot_acc_animation(metrics)
        st.write("### THE DECISION BOUNDARY")
        st.pyplot(plot_decision_boundary(lambda x: predict(params, x.T), X_test, Y_test))
