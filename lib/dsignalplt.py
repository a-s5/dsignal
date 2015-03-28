import matplotlib.pyplot as plt
import lib.dsignal as ds
import numpy as np

def pltSignal(x, X = []):
    """ Narisi signal in amplitudni ter fazni spekter
    """
    if not X:
        X = ds.DFT(x)
    A = np.absolute(X)
    phi = np.angle(X)
    N = np.size(x)
    n = np.arange(N)

    plt.subplot(3, 1, 1)
    plt.plot(n, x, 'b.')
    plt.title("Diskretni casovni prostor")
    plt.xlabel('n'), plt.ylabel('signal')

    plt.subplot(3, 1, 2)
    plt.stem(n, A, markerfmt='r.', linefmt='r')
    plt.title("Frekvenci prostor")
    plt.xlabel('k'), plt.ylabel('amplituda')

    plt.subplot(3, 1, 3)
    plt.plot(n, phi, 'g.')
    plt.title("Frekvenci prostor")
    plt.xlabel('k'), plt.ylabel('faza')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


def pltConvolution(x, y, K = []):
    """ Narisi oba signala in njuno konvolucijo
    """
    N = np.size(x)
    n = np.arange(N)
    if not K:
        K = ds.convolve(x, y)

    plt.subplot(2, 1, 1)
    plt.plot(n, x, 'r.', y, 'b.')
    plt.xlabel('n'), plt.ylabel('signal')

    plt.subplot(2, 1, 2)
    plt.plot(n, K, 'g.')
    plt.xlabel('n'), plt.ylabel('konvolucija')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

