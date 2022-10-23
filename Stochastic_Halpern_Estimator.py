import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def grad_est_x(x, y, A, b, theta, toss, batch, x_prev=None, y_prev=None, grad_est_prev=None):
    batch_size = len(batch)
    A_batch = A[batch, ]
    y_batch = y[batch]
    if toss == 1:
        grad_x = A_batch.T.dot(A_batch.dot(x) - y_batch) / batch_size
    else:
        y_batch_prev = y_prev[batch]
        grad_diff = ((A_batch.dot(x) - y_batch).dot(A_batch) - (A_batch.dot(x_prev) - y_batch_prev).dot(A_batch)) / batch_size
        grad_x = grad_est_prev + grad_diff
    return grad_x

def grad_est_y(x, y, A, b, theta, toss, batch, x_prev=None, y_prev=None, grad_est_prev=None):
    batch_size = len(batch)
    A_batch = A[batch, ]
    y_batch = y[batch]
    grad_y = np.zeros(len(y))
    if toss == 1:
        grad = (y_batch - A_batch.dot(x)) / batch_size
        grad_reg = - theta * (y - b) / batch_size
        grad += grad_reg[batch]
        grad_y[batch] = - grad
    else:
        y_batch_prev = y_prev[batch]
        grad = (y_batch - A_batch.dot(x)) / batch_size
        grad_prev = (y_batch_prev - A_batch.dot(x_prev)) / batch_size
        grad_reg = - theta * (y - b) / batch_size
        grad_reg_prev = - theta * (y_prev - b) / batch_size
        grad += grad_reg[batch]
        grad_prev += grad_reg_prev[batch]
        grad_y[batch] = grad_est_prev[batch] - (grad - grad_prev)
    return grad_y

def grad_norm(x, y, A, b, theta):
    N = len(b)
    grad_x = A.T.dot(A.dot(x) - y) / N
    grad_y = (y - A.dot(x)) / N - theta * (y - b) / N
    grad = np.append(grad_x, grad_y)
    return np.linalg.norm(grad)

def eta_update(eta, k, L):
    M = 9 * (L**2)
    return (1 - 1 / (k + 1)**2 - M * eta**2) * eta * (k + 1)**2 / ((1 - M * eta**2) * k * (k + 2))


if __name__ == "__main__":
    # prepare data and labels
    data = pd.read_csv("./train.csv")
    A = data.iloc[:, :81].to_numpy()
    b = data.iloc[:, -1].to_numpy()
    n = A.shape[0]
    d = A.shape[1]
    for i in range(n):
        A[i, ] /= np.linalg.norm(A[i, ])
        b[i] /= np.linalg.norm(b)
    theta = 1.5

    # single sample
    set_random_seed(11)
    x0 = np.random.rand(d)
    set_random_seed(11)
    y0 = np.random.rand(n)
    vx, vy = x0, y0
    x, y = x0, y0

    s1 = 1
    toss = 1
    batch = np.random.choice(n, s1, replace=False)
    grad_x_curr = grad_est_x(vx, vy, A, b, theta, toss, batch)
    grad_y_curr = grad_est_y(vx, vy, A, b, theta, toss, batch)
    grad_x_prev, grad_y_prev = grad_x_curr, grad_y_curr

    L_x = max(abs(np.linalg.eigvals(A.T.dot(A)) / n))
    L_y = (theta - 1) / n
    eta_x = 1e-3
    eta_y = 1e-3

    res_single = list()
    idx_single = list()
    cnt = s1
    epoch = 0
    k = 1
    res_single.append(grad_norm(x, y, A, b, theta))
    idx_single.append(0)
    while cnt < 30000:
        lam = 1 / (k + 1)
        vx_prev, vy_prev = vx, vy
        vx = lam * x0 + (1 - lam) * x - eta_x * grad_x_prev
        vy = lam * y0 + (1 - lam) * y - eta_y * grad_y_prev
        
        p = 1
        toss = np.random.binomial(1, p)
        if k == 1:
            toss = 1
        if toss == 1:
            batch = np.random.choice(n, s1, replace=False)
            grad_x_curr = grad_est_x(vx, vy, A, b, theta, toss, batch)
            grad_y_curr = grad_est_y(vx, vy, A, b, theta, toss, batch)
            grad_x_prev = grad_x_curr
            grad_y_prev = grad_y_curr
            cnt += s1
        x = lam * x0 + (1 - lam) * x - eta_x * grad_x_prev
        y = lam * y0 + (1 - lam) * y - eta_y * grad_y_prev
        if k % 10 == 0:
            res_single.append(grad_norm(x, y, A, b, theta))
            idx_single.append(cnt)
        eta_x = eta_update(eta_x, k, L_x)
        eta_y = eta_update(eta_y, k, L_x)
        k += 1
    
    # Mini-batch
    set_random_seed(11)
    x0 = np.random.rand(d)
    set_random_seed(11)
    y0 = np.random.rand(n)
    vx, vy = x0, y0
    x, y = x0, y0

    s1 = 32
    toss = 1
    batch = np.random.choice(n, s1, replace=False)
    grad_x_curr = grad_est_x(vx, vy, A, b, theta, toss, batch)
    grad_y_curr = grad_est_y(vx, vy, A, b, theta, toss, batch)
    grad_x_prev, grad_y_prev = grad_x_curr, grad_y_curr

    L_x = max(abs(np.linalg.eigvals(A.T.dot(A)) / n))
    L_y = (theta - 1) / n
    eta_x = 1e-1
    eta_y = 1e-1

    res_batch = list()
    idx_batch = list()
    cnt = s1
    epoch = 0
    k = 1
    res_batch.append(grad_norm(x, y, A, b, theta))
    idx_batch.append(0)
    while cnt < 30000:
        lam = 1 / (k + 1)
        vx_prev, vy_prev = vx, vy
        vx = lam * x0 + (1 - lam) * x - eta_x * grad_x_prev
        vy = lam * y0 + (1 - lam) * y - eta_y * grad_y_prev
        
        p = 1
        toss = np.random.binomial(1, p)
        if k == 1:
            toss = 1
        if toss == 1:
            batch = np.random.choice(n, s1, replace=False)
            grad_x_curr = grad_est_x(vx, vy, A, b, theta, toss, batch)
            grad_y_curr = grad_est_y(vx, vy, A, b, theta, toss, batch)
            grad_x_prev = grad_x_curr
            grad_y_prev = grad_y_curr
            cnt += s1
        x = lam * x0 + (1 - lam) * x - eta_x * grad_x_prev
        y = lam * y0 + (1 - lam) * y - eta_y * grad_y_prev
        if k % 10 == 0:
            res_batch.append(grad_norm(x, y, A, b, theta))
            idx_batch.append(cnt)
        eta_x = eta_update(eta_x, k, L_x)
        eta_y = eta_update(eta_y, k, L_x)
        k += 1
    
    # PAGE
    set_random_seed(11)
    x0 = np.random.rand(d)
    set_random_seed(11)
    y0 = np.random.rand(n)
    vx, vy = x0, y0
    x, y = x0, y0

    s1 = 64
    s2 = 2
    toss = 1
    batch = np.random.choice(n, s1, replace=False)
    grad_x_curr = grad_est_x(vx, vy, A, b, theta, toss, batch)
    grad_y_curr = grad_est_y(vx, vy, A, b, theta, toss, batch)
    grad_x_prev, grad_y_prev = grad_x_curr, grad_y_curr

    L_x = max(abs(np.linalg.eigvals(A.T.dot(A)) / n))
    L_y = (theta - 1) / n
    eta_x = 5e-2
    eta_y = 5e-2

    res = list()
    idx = list()
    cnt = s1
    epoch = 0
    k = 1
    res.append(grad_norm(x, y, A, b, theta))
    idx.append(0)
    while cnt < 30000:
        lam = 1 / (k + 1)
        vx_prev, vy_prev = vx, vy
        vx = lam * x0 + (1 - lam) * x - eta_x * grad_x_prev
        vy = lam * y0 + (1 - lam) * y - eta_y * grad_y_prev
        
        p = 2 / (k + 1)
        s1 = k if k >= s1 else s1
        toss = np.random.binomial(1, p)
        if k == 1:
            toss = 1
        if toss == 1:
            batch = np.random.choice(n, s1, replace=False)
            grad_x_curr = grad_est_x(vx, vy, A, b, theta, toss, batch)
            grad_y_curr = grad_est_y(vx, vy, A, b, theta, toss, batch)
            grad_x_prev = grad_x_curr
            grad_y_prev = grad_y_curr
            cnt += s1
        else:
            batch = np.random.choice(n, s2, replace=False)
            grad_x_curr = grad_est_x(vx, vy, A, b, theta, toss, batch, x_prev=vx_prev, y_prev=vy_prev, grad_est_prev=grad_x_prev)
            grad_y_curr = grad_est_y(vx, vy, A, b, theta, toss, batch, x_prev=vx_prev, y_prev=vy_prev, grad_est_prev=grad_y_prev)
            grad_x_prev = grad_x_curr
            grad_y_prev[batch] = grad_y_curr[batch]
            cnt += s2
        x = lam * x0 + (1 - lam) * x - eta_x * grad_x_prev
        y = lam * y0 + (1 - lam) * y - eta_y * grad_y_prev
        if k % 10 == 0:
            res.append(grad_norm(x, y, A, b, theta))
            idx.append(cnt)
        eta_x = eta_update(eta_x, k, L_x)
        eta_y = eta_update(eta_y, k, L_x)
        k += 1

    # Plot
    # plt.title("halpern_sgd")
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams["figure.figsize"] = (12,8)
    markers = ["v","^","<",">","o","s","p","P","*"]

    plt.plot(idx_single, np.array(res_single), "-", color="teal", linewidth=3.0, label="Single Sample", marker=markers[2], markersize=35, markevery=400, fillstyle="none", markeredgewidth=3)
    plt.plot(idx_batch, np.array(res_batch), "-", color="orange", linewidth=3.0, label="Minibatch", marker=markers[3], markersize=35, markevery=10, fillstyle="none", markeredgewidth=3)
    plt.plot(idx, np.array(res), "-", color="dodgerblue", linewidth=3, label="PAGE", marker=markers[0], markersize=25, markevery=70, fillstyle="none", markeredgewidth=3)
    
    plt.yscale("log")
    plt.xlabel("\#grad", fontsize=50)
    plt.ylabel(r"$\| F(\mathbf{u})\|$", fontsize=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=45)
    plt.ylim()
    plt.legend(loc='upper right', fontsize=25)
    plt.savefig("RLS_super_estimator.png", bbox_inches='tight')
    plt.show()