############### THE PLOTLY ANIMATION GENERATOR AND THE DECISION BOUNDARY GENERATOR ###############

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def plot_loss_animation(df):
    min_loss = (min(min(df['LOSS_TRAIN']), min(df['LOSS_VAL'])))
    max_loss = (max(max(df['LOSS_TRAIN']), max(df['LOSS_VAL'])))

    fig = go.Figure(
        layout=go.Layout(
            updatemenus=[dict(type="buttons", direction="left", x= 0.5, y= - 0.1), ],
            xaxis=dict(range=["0", str(len(list(df['EPOCHS'])) * 10)],
                       autorange=False, tickwidth=2,
                       title_text=".....................Epochs per 10"),
            yaxis=dict(range=[min_loss, max_loss],
                       autorange=False,
                       title_text="Loss of the neural net"),
            title="LOSS OF NEURAL NET (TRAIN AND THE DEV SETS)",
        ))

    init = 1

    fig.add_trace(
        go.Scatter(x=df.EPOCHS[:init],
                   y=df.LOSS_TRAIN[:init],
                   name="LOSS_TRAIN",
                   visible=True,
                   line=dict(width = 2, color="DarkSlateGrey")))

    fig.add_trace(
        go.Scatter(x=df.EPOCHS[:init],
                   y=df.LOSS_VAL[:init],
                   name="LOSS_VAL",
                   visible=True,
                   line=dict(width = 2, color="red")))

    fig.update(frames=[
        go.Frame(
            data=[
                go.Scatter(x=df.EPOCHS[:k], y=df.LOSS_TRAIN[:k]),
                go.Scatter(x=df.EPOCHS[:k], y=df.LOSS_VAL[:k])]
        )
        for k in range(init, len(df) + 1)])

    fig.update_xaxes(ticks="outside", tickwidth=1, tickcolor='white', ticklen=5)
    fig.update_yaxes(ticks="outside", tickwidth=1, tickcolor='white', ticklen=1)
    fig.update_layout(yaxis_tickformat=',')
    fig.update_layout(legend=dict(x=0, y=1.1), legend_orientation="h")

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 100}}]),
                    dict(label="TRAINING",
                         method="update",
                         args=[{"visible": [False, True]},
                               {"showlegend": True}]),
                    dict(label="VALIDATION",
                         method="update",
                         args=[{"visible": [True, False]},
                               {"showlegend": True}]),
                    dict(label="All",
                         method="update",
                         args=[{"visible": [True, True, True]},
                               {"showlegend": True}]),
                ]))])

    st.plotly_chart(fig)


def plot_acc_animation(df):
    min_acc = (min(min(df['ACC_TRAIN']), min(df['ACC_VAL'])))
    max_acc = (max(max(df['ACC_TRAIN']), max(df['ACC_VAL'])))

    fig1 = go.Figure(
        layout=go.Layout(
            updatemenus=[dict(type="buttons", direction="left", x=0.5, y= -0.1), ],
            xaxis=dict(range=["0", str(len(list(df['EPOCHS'])) * 10)],
                       autorange=False, tickwidth=0.5,
                       title_text=".....................Epochs per 10"),
            yaxis=dict(range=[min_acc, 1],
                       autorange=False,
                       title_text="Accuracy of the neural net"),
            title="ACC OF NEURAL NET (TRAIN AND THE DEV SETS)",
        ))

    init = 1

    fig1.add_trace(
        go.Scatter(x=df.EPOCHS[:init],
                   y=df.ACC_TRAIN[:init],
                   name="ACC_TRAIN",
                   visible=True,
                   line=dict(width = 2, color="DarkSlateGrey")))

    fig1.add_trace(
        go.Scatter(x=df.EPOCHS[:init],
                   y=df.ACC_VAL[:init],
                   name="ACC_VAL",
                   visible=True,
                   line=dict(width = 2, color="red")))

    fig1.update(frames=[
        go.Frame(
            data=[
                go.Scatter(x=df.EPOCHS[:k], y=df.ACC_TRAIN[:k]),
                go.Scatter(x=df.EPOCHS[:k], y=df.ACC_VAL[:k])]
        )
        for k in range(init, len(df) + 1)])

    fig1.update_xaxes(ticks="outside", tickwidth=1, tickcolor='white', ticklen=5)
    fig1.update_yaxes(ticks="outside", tickwidth=1, tickcolor='white', ticklen=1)
    fig1.update_layout(yaxis_tickformat=',')
    fig1.update_layout(legend=dict(x=0, y=1.1), legend_orientation="h")

    fig1.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 100}}]),
                    dict(label="TRAINING",
                         method="update",
                         args=[{"visible": [False, True]},
                               {"showlegend": True}]),
                    dict(label="VALIDATION",
                         method="update",
                         args=[{"visible": [True, False]},
                               {"showlegend": True}]),
                    dict(label="All",
                         method="update",
                         args=[{"visible": [True, True, True]},
                               {"showlegend": True}]),
                ]))])

    st.plotly_chart(fig1)


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    return plt


######################## THESE ANIMATION CODES ARE DONE WITH  A GREAT HELP OF STACK OVERFLOW ########################
