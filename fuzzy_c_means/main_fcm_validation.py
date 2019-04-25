# Artur Mello
# Fuzzy C Means - Algorithm validation and performance analysis
# TP 1 - Sistemas Nebulosos
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from fuzzy_c_means import fuzzy_c_means


def main():
    k = 4
    samples = np.asarray(io.loadmat("fcm_dataset.mat")["x"])
    avg_iterations = 0
    reps = 100

    for i in range(reps):

        samples, centroids, data_clusters, iterations = fuzzy_c_means(samples, k)
        avg_iterations += iterations

    plt.scatter(samples[:,0], samples[:, 1], c=data_clusters[:, 0])
    plt.scatter(centroids[:,0], centroids[:, 1], c='red')
    plt.title('Amostras Categorizadas')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('teste_fcm.png')
    plt.show()
    print("Convergência alcançada, em média, em {} iterações".format(avg_iterations/reps))


if __name__ == "__main__":
    main()
