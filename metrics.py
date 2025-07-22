import numpy as np


def load_iq_csv(path: str) -> np.ndarray:
    """Load CSV file with columns I and Q into a complex numpy array."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float32)
    return data[:, 0] + 1j * data[:, 1]


def nmse(reference: np.ndarray, output: np.ndarray) -> float:
    """Return Normalized Mean Square Error between reference and output in dB."""
    reference = np.asarray(reference).astype(np.complex64)
    output = np.asarray(output).astype(np.complex64)
    if reference.shape != output.shape:
        raise ValueError("reference and output must have the same shape")
    error_power = np.mean(np.abs(reference - output) ** 2)
    ref_power = np.mean(np.abs(reference) ** 2)
    if ref_power == 0:
        raise ValueError("reference signal power is zero")
    nmse_linear = error_power / ref_power
    return 10 * np.log10(nmse_linear)


def evm(reference: np.ndarray, output: np.ndarray) -> float:
    """Return RMS Error Vector Magnitude between reference and output in %."""
    reference = np.asarray(reference).astype(np.complex64)
    output = np.asarray(output).astype(np.complex64)
    if reference.shape != output.shape:
        raise ValueError("reference and output must have the same shape")
    error_power = np.mean(np.abs(reference - output) ** 2)
    ref_power = np.mean(np.abs(reference) ** 2)
    if ref_power == 0:
        raise ValueError("reference signal power is zero")
    evm_linear = error_power / ref_power
    evm_percent = 100 * np.sqrt(evm_linear)
    evm_db = 20 * np.log10(evm_percent / 100.0)
    return evm_db

if __name__ == '__main__':
    ref = load_iq_csv('ideal_input.csv')  # or DPD input
    out = load_iq_csv('DPD_and_PA_output.csv')  # measured output
    print('NMSE:', nmse(ref, out))
    print('EVM:', evm(ref, out))
