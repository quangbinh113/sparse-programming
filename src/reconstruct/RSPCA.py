# Import packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib import cm, colors
import cv2
import matplotlib.pyplot as plt

'''
Sparse PCA
'''

class SparsePCA():

    def __init__(self, n_components, alpha=100, verbose=False):
        self.n_components = n_components
        self.alpha = alpha
        self.verbose = verbose
        self.explained_variance = None

    def run_regular_pca_(self, X):

        u, s, vt = cp.linalg.svd(X)
        v = cp.squeeze(vt[0, :])
        if self.verbose:
            print(v @ cp.cov(X.T) @ v)
            print("l1 norm of unconstrained component:", cp.linalg.norm(v, 1))
        return v

    ### Methods for gradient calculation. Don't need to be called outside thise file!
    def abs(self, w):
        return w * cp.tanh(self.alpha * w)

    def abs_prime(self, w):
        diag = cp.tanh(self.alpha * w) + self.alpha * w * (1 - cp.tanh(self.alpha * w) ** 2)
        return diag

    def u1(self, X, w):
        return w @ cp.cov(X.T) @ w

    def u2(self, w):
        return - cp.sum(self.abs(w))

    def v1(self, w):
        return cp.linalg.norm(w) ** 2

    def v2(self, w):
        return cp.linalg.norm(w) ** 2

    def u1_prime(self, X, w):
        u_prime = 2 * w.T @ cp.cov(X.T)
        return u_prime

    def u2_prime(self, w):
        return - self.abs_prime(w)

    def v1_prime(self, w):
        return 2 * w

    def v2_prime(self, w):
        return 2 * w

    def compute_first_grad(self, X, w):
        enum = self.u1_prime(X, w) * self.v1(w) - self.u1(X, w) * self.v1_prime(w)
        denom = self.v1(w) ** 2
        return enum / denom

    def compute_reg_grad(self, w):
        enum = self.u2_prime(w) * self.v2(w) - self.u2(w) * self.v2_prime(w)
        denom = self.v2(w) ** 2
        grad = enum / denom
        return grad

    def compute_grad(self, X, w, lambd):
        grad = self.compute_first_grad(X, w) + lambd * self.compute_reg_grad(w)
        # Orthogonalize
        grad = grad - (cp.dot(grad, w) / cp.dot(w, w)) * w
        return grad

    def compute_loss(self, X, w, lambd):
        return self.u1(X, w) / self.v1(w) + lambd * self.u2(w) / self.v2(w)

    def sparsify_w(self, X, w, lambd, lr, steps):

        if self.verbose == True:
            print("starting loss:", self.compute_loss(X, w, lambd))
        for step in range(steps):
            grad = self.compute_grad(X, w, lambd)
            w += lr * grad
            w = w / cp.linalg.norm(w)
            if step % 1000 == 0 and self.verbose == True:
                print("step:", step)
                print("grad norm:", cp.linalg.norm(grad))
                print("loss:", self.compute_loss(X, w, lambd))
                print("var loss:", self.u1(X, w) / self.v1(w), "reg loss:", lambd * self.u2(w) / self.v2(w))
                print()
        return w

    def find_component(self, X, lambd, lr, steps):

        w = self.run_regular_pca_(X)
        w_out = self.sparsify_w(X, w, lambd, lr, steps)
        return w_out

    def remove_variance(self, X, w):

        X = X - X @ cp.outer(w, w)
        return X

    def fit(self, X, lambdas, lr=1e-6, steps=1000):

        # Keep track of the original X
        X_original = cp.copy(X)
        if not isinstance(lambdas, list):
            lambdas = self.n_components * [lambdas]
        # Initialize component matrix
        W = cp.zeros((X.shape[1], self.n_components))
        # Loop over number of desired components
        for component_id in range(self.n_components):
            print("Computing component", component_id, "...")
            w = self.find_component(X, lambdas[component_id], lr, steps)
            W[:, component_id] = w
            # Remove the variance of the found component from the data to find the next one
            X = self.remove_variance(X, w)
        # Compute explained variance
        self.compute_explained_variance(X_original, W)
        return W

    def compute_explained_variance(self, X, W):

        self.explained_variance = cp.diag(W.T @ cp.cov(X.T) @ W) / cp.trace(cp.cov(X.T))
        return None


'''
Robust Sparse PCA
'''

class Robust_Sparse_PCA:

    def __init__(self, imax=1000, fthres=1e-7, verbose=False):
        self.imax = imax  # Maximum number of iterations
        self.fthres = fthres  # Convergence threshold
        self.verbose = verbose  # Print progress during iterations

    def shrink(self, X, tau):
        Y = np.abs(X) - tau
        return np.sign(X) * np.maximum(Y, np.zeros_like(Y))

    def SVT(self, X, tau):
        U, S, VT = np.linalg.svd(X, full_matrices=0)
        out = U @ np.diag(self.shrink(S, tau)) @ VT
        return out

    def fit(self, X):
        m, n = X.shape  # Store matrix dimensions
        mu = m * n / (4 * np.sum(np.abs(X.reshape(-1))))  # Calculate mu
        lambd = 1 / np.sqrt(np.max(np.asarray([m, n])))  # Calculate lambda
        thres = self.fthres * np.linalg.norm(X)  # Determine threshold (tolerance)

        # Initialize matrices L, S, and Y
        L = np.zeros_like(X)
        S = np.zeros_like(X)
        Y = np.zeros_like(X)

        count = 0
        # Continue iterating while X ≈ L + S is above threshold or maximum number of iterations has been reached
        while (np.linalg.norm(X - L - S) > thres) and (count < self.imax):
            if self.verbose:
                print(f'{np.round(thres / np.linalg.norm(X - L - S), 3)}% ...   ({count}/{self.imax})')

            # Update L, S, and Y
            L = self.SVT(X - S + 1 / mu * Y, 1 / mu)
            S = self.shrink(X - L + 1 / mu * Y, lambd / mu)
            Y = Y + mu * (X - L - S)
            count += 1

        # Print number of iterations required
        if self.verbose:
            if count == self.imax:
                print(f'No convergence after {count} iterations')
            else:
                print(f'Solution found after {count} iterations')

        self.L_ = L
        self.S_ = S

        return self


'''
Noise models
'''

class Noise_Models:
    def __init__(self):
        pass

    @staticmethod
    def to_std_float(img):
        img.astype(np.float16, copy=False)
        img = np.multiply(img, (1/255))
        return img

    @staticmethod
    def to_std_uint8(img):
        img = cv2.convertScaleAbs(img, alpha=(255/1))
        return img

    @staticmethod
    def display_result(img, title='Image', show=True):
        fig, axs = plt.subplots(figsize=(7, 12))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        if show:
            plt.tight_layout()
            plt.show()

    @staticmethod
    def salt_n_pepper(img, pad=101, show=True):
        img = Noise_Models.to_std_float(img)
        noise = np.random.randint(pad, size=(img.shape[0], img.shape[1], 1))
        img = np.where(noise == 0, 0, img)
        img = np.where(noise == (pad-1), 1, img)
        img = Noise_Models.to_std_uint8(img)
        if show:
            Noise_Models.display_result(img, 'Image with Salt & Pepper Noise')

        return img

    @staticmethod
    def add_gaussian_noise(img, mean=0, std=0.1, show=True):
        img = Noise_Models.to_std_float(img)
        noise = np.random.normal(mean, std, size=(img.shape[0], img.shape[1], img.shape[2]))
        img_with_noise = img + noise
        img_with_noise = np.clip(img_with_noise, 0, 1)
        img_with_noise = Noise_Models.to_std_uint8(img_with_noise)
        if show:
            Noise_Models.display_result(img_with_noise, 'Image with Gaussian Noise')

        return img_with_noise
