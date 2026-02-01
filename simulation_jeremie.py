import numpy as np
import matplotlib.pyplot as plt


# Bit source generation

def source_bit(N, p):
    """(int, float) -> np.array
    Simulate a binary source that generates N bits, where each bit is 1 with probability p.
    """
    return np.array([np.random.binomial(1, p) for _ in range(N)])
!ù
m^$ùùdef prbs(array):
    """(np.array) -> np.array
    Convert a binary array (0,1) into bipolar format (-1,1).
    """
    return 2 * array - 1

N = int(2**10) #length of the bit sequence
p = 0.5 # probability of having 1
M = 16 # M number of the QAM constellation
bit_source = source_bit(N, p)

print("Bit source output length : ", len(bit_source),f"Probability to have 1 (probability set to {p}", sum([1 for bit in bit_source if bit == 1])/N)

symbol_source = None

def compute_energy(symbols):
    """(np.array) --> float
    Compute the average energy of a given constellation."""
    # convert the matrix to vector
    symbols = np.array(symbols).flatten()
    energy = 0
    n = len(symbols)
    for symbol in symbols:
        energy += abs(symbol)**2/n
    return energy

def generate_constellation(M, energy = 1):
    """(int, float) --> (np.array, float)
    Generate a square M-QAM constellation normalized to the specified energy."""

    n = int(np.log2(M))

    constellation = []
    constellation_matrix = np.zeros((n,n), dtype = 'complex') # MOVE TO MATRIX REPRESENTATION



    a = np.arange(n*2, step =2) - n + 1 # to be symmetric around 0
    b = np.arange(n*2, step = 2) - n + 1

    for l in range(n):
        for m in range(n):
            symbol = a[l] + 1j * b[m]
            constellation.append(symbol)
            constellation_matrix[l][m] = symbol

    constellation_energy = compute_energy(constellation)
    print(f"Energy of the generated constellation ({M}-QAM) before normalization : ", constellation_energy)

    return constellation_matrix/np.sqrt(constellation_energy/energy)


constellation_QAM = generate_constellation(M)
print(f"Energy of the generated constellation ({M}-QAM): ", compute_energy(constellation_QAM))

plt.figure()
plt.plot(np.real(constellation_QAM.flatten()), np.imag(constellation_QAM.flatten()), 'x')
plt.grid()
plt.show()

def generate_mapping(constellation):
    """(np.array) --> dict
    Generate a mapping dictionary from bit sequences to constellation symbols.
    """
    dico = {}
    flat_constellation = constellation.flatten()
    n = len(flat_constellation)
    for i in range(n):
        dico[str(np.binary_repr(i, width=4))] = flat_constellation[i]

    return dico

mapping_QAM = generate_mapping(constellation_QAM)
print(mapping_QAM)


def mapper(bit_array, constellation, mapping):
    """(np.array, np.array, dict) --> np.array
    Map the input bit array to constellation symbols using the provided mapping.
    """
    flat_constellation = constellation.flatten()
    M = len(flat_constellation)
    n = int(np.log2(M))

    list_symbols = []

    for i in range(0, len(bit_array), n):
        bit_segment = bit_array[i:i+n]
        bit_str = ''.join(str(bit) for bit in bit_segment)

        symbol = mapping[bit_str]
        list_symbols.append(symbol)

    return np.array(list_symbols)

# BE CAREFUL, THE MAPPING IS NOT CORRECT YET

symbol_source = mapper(bit_source, constellation_QAM, generate_mapping(constellation_QAM))




def inner_product(x, y, dt=1):
    """(np.array, np.array) --> complex
    Compute the inner product between two complex vectors x and y.
    """
    return np.sum(np.conj(x) * y) * dt

bandwith = 3
T = 1 / bandwith
time = np.linspace(-10*T, 10*T, 200001)

def demod(time, bandwidth):
    """(float, np.array, float) --> np.array
    Demap the received signal back to bit sequences based on the constellation.
    """

    T = 1 / bandwidth
    dt = time[1] - time[0]

    s_0 = 1
    s_1 = 1 + 1j
    s_list = [s_0, s_1]



    x = np.zeros(len(time), dtype='complex')

    # construct the signal
    for k, s in enumerate(s_list):
        x += s * np.sinc((time - k*T) / T)

    r = []
    # compute the received symbols
    for l in range(len(s_list)):
        pulse = np.sinc((time - l*T) / T)
        r_l = bandwidth * inner_product(x, pulse, dt)
        r.append(r_l)

    print(f"Received symbol, r: {r}")

    return np.array(r)

#demapped_bits = demod(time, bandwidth=3)


def demod(time, signal, bandwidth):
    """
    Recover symbols using inner product.
    time: np.array, time vector (must be long enough)
    signal: np.array, input symbols
    bandwidth: float
    """
    T = 1 / bandwidth
    dt = time[1] - time[0]

    N_symbols = len(signal)

    # Construct the signal
    x = np.zeros_like(time, dtype=complex)
    for k, s in enumerate(signal):
        x += s * np.sinc((time - k*T) / T)

    # Recover symbols via inner product
    r = []
    for k in range(N_symbols):
        pulse = np.sinc((time - k*T) / T)
        r_k = (1/T) * np.sum(np.conj(pulse) * x) * dt  # inner product
        r.append(r_k)

    return np.array(r)

#demapped_bits = demod(time, bandwidth=3)

# Simulation parameters
dt = 0.001  # Time step
z = 5  # Propagation step

# Dispersion parameter
beta2 = 1

# time vector for the pulse
time_2 = np.arange(-10, 10, dt)

# Initial Gaussian pulse parameters

A = 1.0  # Amplitude
sigma = 2 # Width parameter

q = A * np.exp(- (time_2**2) / (2 * sigma**2))  # Initial Gaussian pulse

def linear_step(q, beta2, z, w):
    """(np.array, float, float, np.array) --> np.array
    Apply the linear dispersion step in the frequency domain.
    q : input signal
    beta2 : dispersion parameter
    dz : propagation step
    w : angular frequency vector
    """
    dispersion = np.exp(-0.5j * beta2 * (w**2) * z)
    q_hat = np.fft.fft(q)
    q_hat *= dispersion
    return np.fft.ifft(q_hat)

def nonlinear_step(q, gamma, z):
    """(np.array, float, float) --> np.array
    Apply the nonlinear step in the time domain.
    q : input signal
    gamma : nonlinearity parameter
    dz : propagation step
    """
    return q * np.exp(1j * gamma * np.abs(q)**2 * z)


def propagation(q, beta2, gamma, z, dt, N):
    """(np.array, float, float, float, float) --> np.array
    Perform one step of propagation using the split-step Fourier method.
    q : input signal
    beta2 : dispersion parameter
    gamma : nonlinearity parameter
    dz : propagation step
    dt : time step
    """
    for _ in range(N):
        q = nonlinear_step(q, gamma, z / 2)
        q = linear_step(q, beta2, z / 2, 2 * np.pi * np.fft.fftfreq(len(q), dt))
    return q


plt.figure()
# time domain plots
plt.subplot(1,2,1)
# initial
plt.plot(time_2, np.abs(q)**2, label='Initial Pulse')
w = 2 * np.pi * np.fft.fftfreq(len(time_2), dt)
q_after_linear = linear_step(q, beta2, z / 2, w)

# after linear step
plt.plot(time_2, np.abs(q_after_linear)**2, label='After Linear Step')

# after non linear step

plt.plot(time_2, np.abs(nonlinear_step(q, gamma=1, z=z))**2, label='After Nonlinear Step')

# after propagation
q_after_propagation = propagation(q, beta2, gamma=1, z=z, dt=dt, N=1)
plt.plot(time_2, np.abs(q_after_propagation)**2, label='After Propagation')

plt.xlabel('Time')
plt.ylabel('Intensity |q|^2')
plt.title('Effect of Linear Dispersion Step on Gaussian Pulse')
plt.legend()
plt.tight_layout()
plt.grid()

# frequency domain plots
plt.subplot(1,2,2)
freq = np.fft.fftfreq(len(time_2), dt)
plt.plot(freq, np.abs(np.fft.fft(q))**2, label='Initial Pulse')
plt.plot(freq, np.abs(np.fft.fft(q_after_linear))**2, label='After Linear Step')
plt.plot(freq, np.abs(np.fft.fft(nonlinear_step(q, gamma=1, z=z)))**2, label='After Nonlinear Step')
plt.plot(freq, np.abs(np.fft.fft(q_after_propagation))**2, label='After Propagation')
plt.xlim(-1.5, 1.5)
plt.xlabel('Frequency')
plt.ylabel('Spectral Intensity')
plt.title('Effect of Linear Dispersion Step in Frequency Domain')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


















