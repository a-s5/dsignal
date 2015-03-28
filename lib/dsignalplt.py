import matplotlib.pyplot as plt
import lib.dsignal as ds
import numpy as np

def pltSignal(x, *args, name=''):
    """ Narisi signal in amplitudni ter fazni spekter
    """
    f = plt.figure(name) if name else plt.figure()
    if args:
        X = args[0]
    else:
        X = ds.DFT(x)

    A = np.absolute(X)
    phi = ds.phase(X)
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
    plt.show(block=False)


def pltConvolution(x, y, *args, name=''):
    """ Narisi oba signala in njuno konvolucijo
    """
    f = plt.figure(name) if name else plt.figure()
    N = np.size(x)
    n = np.arange(N)
    if args:
        K = args[0]
    else:
        K = ds.convolve(x, y)

    plt.subplot(2, 1, 1)
    plt.plot(n, x, 'r.', y, 'b.')
    plt.xlabel('n'), plt.ylabel('signal')

    plt.subplot(2, 1, 2)
    plt.plot(n, K, 'g.')
    plt.xlabel('n'), plt.ylabel('konvolucija')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show(block=False)


def pltCorr(x, y, *args, name=''):
    """ Narisi oba signala in njuno korelacijsko funkcijo
    """
    f = plt.figure(name) if name else plt.figure()
    N = np.size(x)
    n = np.arange(N)
    if args:
        R = args[0]
    else:
        R = ds.fnCorr(x, y)

    plt.subplot(2, 1, 1)
    plt.plot(n, x, 'r.', y, 'b.')
    plt.xlabel('n'), plt.ylabel('signal')

    plt.subplot(2, 1, 2)
    plt.plot(n, R, 'g.')
    plt.xlabel('n'), plt.ylabel('korelacija')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show(block=False)

