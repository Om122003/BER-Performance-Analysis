import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Parameters
N = int(1e4)  # Number of bits (or symbols)
nTx = 2       # Number of transmit antennas

# Transmitter
ip = np.random.rand(1, N) > 0.5  # Generating 0,1 with equal probability
s = 2*ip - 1  # BPSK modulation 0 -> -1, 1 -> 1

Eb_N0_dB = np.arange(0, 42, 2)  # Multiple Eb/N0 values in dB

# Initialize error counters
nErr1 = np.zeros(len(Eb_N0_dB))
nErr2 = np.zeros(len(Eb_N0_dB))

# Loop over different Eb/N0 values
for ii in range(len(Eb_N0_dB)):
    # Noise and Rayleigh channel realization
    n = 1/np.sqrt(2) * (np.random.randn(1, N) + 1j*np.random.randn(1, N))  # AWGN
    h = 1/np.sqrt(2) * (np.random.randn(nTx, N) + 1j*np.random.randn(nTx, N))  # Rayleigh channel

    sr = (1/np.sqrt(nTx)) * np.kron(np.ones((nTx, 1)), s)  # BPSK modulated signal

    # Effective channel for beamforming
    hEff = h * np.exp(-1j * np.angle(h))

    # Received signal (with and without beamforming)
    y1 = np.sum(h * sr, axis=0) + 10**(-Eb_N0_dB[ii]/20) * n  # Without beamforming
    y2 = np.sum(hEff * sr, axis=0) + 10**(-Eb_N0_dB[ii]/20) * n  # With beamforming

    # Equalization (Divide by channel)
    y1Hat = y1 / np.sum(h, axis=0)
    y2Hat = y2 / np.sum(hEff, axis=0)

    # Receiver - hard decision decoding
    ip1Hat = np.real(y1Hat) > 0
    ip2Hat = np.real(y2Hat) > 0

    # Counting the errors
    nErr1[ii] = np.sum(ip != ip1Hat)
    nErr2[ii] = np.sum(ip != ip2Hat)

# Simulated BER
simBer1 = nErr1 / N  # Simulated BER (no beamforming)
simBer2 = nErr2 / N  # Simulated BER (with beamforming)

# Theoretical BER for Rayleigh and AWGN channel
theoryBerAWGN = 0.5 * erfc(np.sqrt(10**(Eb_N0_dB/10)))  # AWGN channel
EbN0Lin = 10**(Eb_N0_dB/10)
theoryBer = 0.5 * (1 - np.sqrt(EbN0Lin / (EbN0Lin + 1)))  # Rayleigh channel

# Theoretical BER for MRC with 2x1 system
p = 1/2 - 1/2 * (1 + 1/EbN0Lin)**(-1/2)
theoryBer_nRx2 = p**2 * (1 + 2 * (1 - p))

# Plotting the results
plt.figure()
plt.semilogy(Eb_N0_dB, theoryBer, 'bp-', label='1x1 (Theory)', linewidth=2)
plt.semilogy(Eb_N0_dB, simBer1, 'rs-', label='2x1 (No beamforming-sim)', linewidth=2)
plt.semilogy(Eb_N0_dB, theoryBer_nRx2, 'gp-', label='2x1 (MRC-theory)', linewidth=2)
plt.semilogy(Eb_N0_dB, simBer2, 'mx-', label='2x1 (Beamforming-sim)', linewidth=2)

plt.axis([0, 35, 1e-5, 0.5])
plt.grid(True)
plt.legend()
plt.xlabel('Eb/No, dB')
plt.ylabel('Bit Error Rate')
plt.title('BER of BPSK modulated signal through Rayleigh channel')
plt.show()
