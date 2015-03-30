import numpy as np

##################################################
# Osnovne oblike signalov
##################################################

def unitStep(M, N):
    """ Stopnica sirine M s periodo N
    """
    return np.concatenate( (np.ones(M), np.zeros(N-M)) )


def sawStep(M, N):
    """ Zagast pulz sirine M s periodo N
    """
    return np.concatenate( (np.linspace(0,1,M), np.zeros(N-M)) )


def sinewave(N, n, phi=0):
    """ Sinus s periodo N zamaknjen za kot phi na intervalu n
    """
    t = np.arange(n)
    return np.sin(2*np.pi*t/N + phi)


def chain(signal, periods, M, N):
    """ Zlepi vec period signala
    """
    x = np.array([])
    for i in range(periods):
        x = np.concatenate( (x, signal(M,N)) )
    return x


def delay(signal, n, cycle=True):
    """ Zakasni signal za n vzorec. Ce je cycle=True, zakasni cikicno
    """
    if cycle:
        return np.concatenate(( signal[n:], signal[0:n]))
    else:
        return np.concatenate(( np.zeros(n), signal[n:]))


##################################################


##################################################
# Znacilne vrednosti
##################################################

def mean(x):
    """ Srednja vrednost signala x
    """
    N = np.size(x)
    return np.sum(x)/N


def meansq(x):
    """ Srednja moc signala x
    """
    return mean(x**2)


def variance(x):
    """ Varianca signala x
    """
    N = np.size(x)
    meanValue = mean(x)
    return np.sum( (x - meanValue)**2 )/N


def corr(x, y):
    """ Korelacija med signaloma x in y
    """
    N = np.size(x)
    if N == np.size(y):
        return np.dot(x,y)/N
    else:
        print("Signala imata razlicno periodo")
        return


def normCorr(x, y):
    """ Normirana korelacija med signaloma x in y
    """
    correlation = corr(x, y)
    return correlation/np.sqrt(meansq(x) * meansq(y))

##################################################


##################################################
# Funkcije in transformi
##################################################

def fnCorr(x, y):
    """ Korelacijska fukncija signalov x in y
    """
    N = np.size(x)
    if N == np.size(y):
        R = np.zeros(N)
        for m in range(N):
            y_tmp = np.concatenate( (y[m:N], y[0:m]) )
            R[m] = np.dot(x, y_tmp)
        return R/N
    else:
        print("Signala imata razlicno periodo")
        return


def fnAutoCorr(x):
    """ Avtokorelacijska funckija signala x
    """
    return fnCorr(x,x)


def convolve(x, y):
    """ Konvolucija signala x in y
    """
    N = np.size(x)
    if N == np.size(y):
        K = np.zeros(N)
        for n in range(N):
            y_tmp = np.concatenate( (y[n::-1], y[N:n:-1]) )
            K[n] = np.dot(x, y_tmp)
        return K
    else:
        print("Signala imata razlicno periodo")
        return


def DFT(x):
    """ Diskretni Fourierov transform signala x
    """
    N = np.size(x)
    k = np.arange(N)
    n = np.ones( (N,N) )

    W = np.exp( -1j*(2*np.pi/N) * ((n*k).T*k) )

    return np.dot(W,x)/N


def IDFT(X):
    """ Inverzni diskretni Fourierov transform spektra X
    """
    N = np.size(X)
    k = np.arange(N)
    n = np.ones( (N,N) )
    
    W = np.exp( 1j*(2*np.pi/N) * ((n*k).T*k) )

    return np.dot(W,X)


def baseW(N, inverse=False):
    """ Vrni matriko baznih vektorjev DFT oz. IDFT
    """
    k = np.arange(N)
    n = np.ones( (N,N) )
    if inverse:
        return np.exp( 1j*(2*np.pi/N) * ((n*k).T*k) )
    else:
        return np.exp( -1j*(2*np.pi/N) * ((n*k).T*k) )


def phase(X):
    """ Fazni spekter spektra X
    """
    Xtmp = np.round(X*10**9)/10**9
    phi = np.angle(Xtmp)
    phi[np.logical_and(Xtmp.imag == 0, Xtmp.real == 0)] = np.NaN
    return phi
