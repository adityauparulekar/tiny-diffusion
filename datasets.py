import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset

def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def square_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = np.ones(n//4)
    y = rng.uniform(-1, 1, n//4)
    X1 = np.stack((x, y), axis=1)
    
    x = -1 * np.ones(n//4)
    y = rng.uniform(-1, 1, n//4)
    X2 = np.stack((x, y), axis=1)
    
    y = np.ones(n//4)
    x = rng.uniform(-1, 1, n//4)
    X3 = np.stack((x, y), axis=1)
    
    y = -1 * np.ones(n//4)
    x = rng.uniform(-1, 1, n//4)
    X4 = np.stack((x, y), axis=1)

    X = np.concatenate((X1, X2, X3, X4)) * 3
    return TensorDataset(torch.from_numpy(X.astype(np.float64)))

def circle_dataset(n=8000, r=0):
    rng = np.random.default_rng(42)
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    x = np.cos(theta)
    y = np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= r
    return TensorDataset(torch.from_numpy(X.astype(np.float64)))
    
def point_1d_dataset(n=8000, r=0):
    x = np.zeros((n, 1))
    return TensorDataset(torch.from_numpy(x.astype(np.float64)))

def dino_dataset(n=8000):
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def get_dataset(name, n=8000):
    if name == "moons":
        return moons_dataset(n)
    elif name == "dino":
        return dino_dataset(n)
    elif name == "line":
        return line_dataset(n)
    elif name == "circle":
        return circle_dataset(n)
    elif name == "square":
        return square_dataset(n)
    elif name == "point1d":
        return point_1d_dataset(n)
    else:
        raise ValueError(f"Unknown dataset: {name}")
