############### THIS IS THE PORTION TO VISUALISE THE TOY DATASET IN A 2D SCATTER PLOT ###############

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_circles, make_moons
import pandas as pd


def plot_decision_boundary(input_dims, model, X, y):
    if input_dims == 2:
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    else:
        print("sorry its not your plot")


def plot_toy_data(type, samples, noise, stream=False):
    if type == 'moons':
        X, Y = make_moons(n_samples=samples, noise=noise)
    else:
        X, Y = make_circles(n_samples=samples, noise=noise)

    if not stream:
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=Y))
        colors = {0: 'red', 1: 'blue'}
        plt.figure(figsize=(8, 8))
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        plt.show()
    else:
        df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=Y))
        colors = {0: 'red', 1: 'blue'}
        plt.figure(figsize=(8, 8))
        fig, ax = plt.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        st.pyplot(fig)
