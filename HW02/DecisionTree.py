import numpy as np
import os
import math
from scipy.stats import entropy as H
import matplotlib.pyplot as plt
import json


def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.num_calls += 1
        return func(*args, **kwargs)

    wrapper.num_calls = 0
    return wrapper

@count_calls
def MakeSubtree(X, y):
    C = []

    # deteremine candidate splits for each feature
    for f in range(len(X[0])):
        C += DetermineCandidateSplits(f, X, y)

    # evaluate stopping criteria
    if (
        len(y) == 0
        or all(S["gain_ratio"] == 0 for S in C)
        or all(S["split_entropy"] == 0 for S in C)
    ): 
        # make a leaf node
        node = {
            # "split": None,
            # "children": None,
            # "data": { "X": X, "y": y },
            # determine class label by majority, default 1
            "predict": 1 if np.sum(y == 1) >= np.sum(y == 0) else 0
        }

    else:

        # find best split
        S = BestSplit(C)

        condition = X[:,S["feature"]] >= S["threshold"]

        X_left = X[condition]
        y_left = y[condition]

        X_right = X[~condition]
        y_right = y[~condition] 

        # make an internal node
        node = {
            "split": S,
            "children":[
                MakeSubtree(X_left, y_left),
                MakeSubtree(X_right, y_right)
            ],
            # "data": None,
            # "predict": None
        }

    return node


def BestSplit(C):
    return max(C, key=lambda S: S['gain_ratio'])

def DetermineCandidateSplits(f, X, y):
    if len(y) == 0: return []

    C = []
    x = X[:,f]
    sorted_idx = x.argsort()

    # compute H(Y)
    p_class = np.bincount(y) / len(y)
    H_class = H(p_class, base=2) # -(p_class[0] * math.log2(p_class[0]) + p_class[1] * math.log2(p_class[1]))

    # print(f"H(Y): {H_class}")

    # loops through adjcent pairs
    for i, j in zip(sorted_idx, sorted_idx[1:]):
        if (y[i] != y[j]):

            # if (x[i] == x[j]): continue

            # threshold
            c = max(x[i], x[j])

            # compute H(Y|S)
            n_split = np.array([[0, 0], [0, 0]])
            for k in range(len(y)):
                if x[k] >= c:
                    if y[k] == 0: n_split[0][0] += 1
                    if y[k] == 1: n_split[0][1] += 1
                else:
                    if y[k] == 0: n_split[1][0] += 1
                    if y[k] == 1: n_split[1][1] += 1

            # num samples that go left
            n_left = n_split[0][0] + n_split[0][1]

            # num samples that go right
            n_right = n_split[1][0] + n_split[1][1]

            H_class_split_left = 0
            if n_left != 0:
                p_class_split_left = n_split[0] / n_left
                H_class_split_left = H(p_class_split_left, base=2)

            H_class_split_right = 0
            if n_right != 0:
                p_class_split_right = n_split[1] / n_right
                H_class_split_right = H(p_class_split_right, base=2)

            p_split = np.array([n_left, n_right]) / len(y)
            H_class_split = p_split[0] * H_class_split_left + p_split[1] * H_class_split_right

            # compute H(S)
            H_split = H(p_split, base=2)

            infoGain = H_class - H_class_split
            gainRatio = 0 if H_split == 0 else infoGain / H_split

            C.append(
                {
                    "feature": f,
                    "threshold": c,
                    "info_gain": infoGain,
                    "gain_ratio": gainRatio,
                    "split_entropy": H_split
                }
            )

            # break

    return C

def VisualizeTree(tree, X, y, title):
    xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 500),
                     np.linspace(-1.5, 1.5, 500))

    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    predictions = np.zeros(len(mesh_data))
    for i, point in enumerate(mesh_data):
        node = tree
        while "predict" not in node:
            if point[node['split']['feature']] >= node["split"]['threshold']:
                node = node['children'][0]
            else:
                node = node['children'][1]
        predictions[i] = node['predict']

    predictions = predictions.reshape(xx.shape)

    plt.contourf(xx, yy, predictions, alpha=0.5, cmap='coolwarm')

    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='.', color='blue', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='.', color='red', label='Class 1')

    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.title(title)
    plt.legend()
    plt.show()

def PrintTree(node, indent):
    if "split" in node:
        print(indent + f"({node['split']['feature']}, {node['split']['threshold']})")
    if "children" in node:
        indent += "    "
        PrintTree(node["children"][0], indent)
        PrintTree(node["children"][1], indent)
    else:
        print(indent + f"Predict {node['predict']}")

def ComputeError(tree, X_test, y_test):
    y_predict = []
    for i, row in enumerate(X_test):
        node = tree
        while "predict" not in node:
            if row[node['split']['feature']] >= node["split"]['threshold']:
                node = node['children'][0]
            else:
                node = node['children'][1]
        y_predict.append(node['predict'])
    
    n_wrong = np.sum(y_predict != y_test)
    n_total = len(y_test)

    return n_wrong / n_total



def Exercise2_7():
    D = np.loadtxt(os.path.abspath("./data/Dbig.txt"))
    X = D[:,:2]
    y = D[:,-1].astype(int)

    n_rows = X.shape[0]
    random_idicies = np.random.permutation(n_rows)

    X_random = X[random_idicies]
    y_random = y[random_idicies]

    X_test = X_random[8192:,:]
    y_test = y_random[8192:]

    X_8192 = X_random[:8192,:]
    y_8192 = y_random[:8192]

    X_2048 = X_random[:2048,:]
    y_2048 = y_random[:2048]

    X_512 = X_random[:512,:]
    y_512 = y_random[:512]

    X_128 = X_random[:128,:]
    y_128 = y_random[:128]

    X_32 = X_random[:32,:]
    y_32 = y_random[:32]

    tree = MakeSubtree(X_8192, y_8192)
    print(MakeSubtree.num_calls)
    print(ComputeError(tree, X_test, y_test))
    # VisualizeTree(tree, X_8192, y_8192, '$D_{8192}$')

    # print(ComputeError(tree, X_test, y_test))
    
    # VisualizeTree(tree, X_2048, y_2048)
    # print(MakeSubtree.num_calls)


    # generate learning curve
    # n = [32, 128, 512, 2048, 8192]
    # err = [
    #     ComputeError(MakeSubtree(X_32, y_32), X_test, y_test),
    #     ComputeError(MakeSubtree(X_128, y_128), X_test, y_test),
    #     ComputeError(MakeSubtree(X_512, y_512), X_test, y_test),
    #     ComputeError(MakeSubtree(X_2048, y_2048), X_test, y_test),
    #     ComputeError(MakeSubtree(X_8192, y_8192), X_test, y_test)
    # ]

    # plt.plot(n, err)
    # plt.xlabel('$n$')
    # plt.ylabel('$err$')
    # plt.title('$n$ vs $err$')
    # plt.legend()
    # plt.show()

def Exercise3():
    from sklearn import tree
    from sklearn.metrics import accuracy_score

    D = np.loadtxt(os.path.abspath("./data/Dbig.txt"))
    X = D[:,:2]
    y = D[:,-1].astype(int)

    n_rows = X.shape[0]
    random_idicies = np.random.permutation(n_rows)

    X_random = X[random_idicies]
    y_random = y[random_idicies]

    X_test = X_random[8192:,:]
    y_test = y_random[8192:]

    X_8192 = X_random[:8192,:]
    y_8192 = y_random[:8192]

    X_2048 = X_random[:2048,:]
    y_2048 = y_random[:2048]

    X_512 = X_random[:512,:]
    y_512 = y_random[:512]

    X_128 = X_random[:128,:]
    y_128 = y_random[:128]

    X_32 = X_random[:32,:]
    y_32 = y_random[:32]

    # tree = tree.DecisionTreeClassifier()
    # tree = tree.fit(X_8192, y_8192)
    # print(tree.tree_.node_count)
    # print(1 - accuracy_score(y_test, tree.predict(X_test)))

    plt.plot(
        [32, 128, 512, 2048, 8192], 
        [
            0.13440265486725667,
            0.08738938053097345,
            0.04646017699115046,
            0.029314159292035402,
            0.011615044247787587
        ])
    plt.xlabel('$n$')
    plt.ylabel('$err$')
    plt.title('$n$ vs $err$')
    plt.legend()
    plt.show()


X = np.array([
    [0 ,0],
    [0, 1],
    [1, 0],
    [1, 1],
])

y = np.array([
    0,
    1,
    1,
    0,
])

# tree = MakeSubtree(X, y)
# print(json.dumps(tree, indent=4))

# PrintTree(tree, "")


plt.scatter(X[:,0], X[:, 1], c=y)
plt.show()