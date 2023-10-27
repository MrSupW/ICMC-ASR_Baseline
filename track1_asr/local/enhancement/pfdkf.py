import os
import glob
import argparse
import numpy as np
import soundfile as sf

from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft


class PFDKF:
    def __init__(self, N, M, A=0.999, P_initial=1, keep_m_gate=0.5, res=False):
        """Initial state of partitioned block based frequency domain kalman filter

        Args:
            N (int): Num of blocks.
            M (int): Filter length in one block.
            A (float, optional): Diag coeff, if more nonlinear components,
            can be set to 0.99. Defaults to 0.999.
            P_initial (int, optional): About the begining covergence. 
            Defaults to 10.
            keep_m_gate (float, optional): When more linear,
            can be set to 0.2 or less. Defaults to 0.5.
        """
        # M = 2*V
        self.N = N
        self.M = M
        self.A = A
        self.A2 = A ** 2
        self.m_smooth_factor = keep_m_gate
        self.res = res

        self.x = np.zeros(shape=(2 * self.M), dtype=np.float32)
        self.m = np.zeros(shape=(self.M + 1), dtype=np.float32)
        self.P = np.full((self.N, self.M + 1), P_initial)
        self.X = np.zeros((self.N, self.M + 1), dtype=complex)
        self.H = np.zeros((self.N, self.M + 1), dtype=complex)
        self.mu = np.zeros((self.N, self.M + 1), dtype=complex)
        self.half_window = np.concatenate(([1] * self.M, [0] * self.M))

    def filt(self, x, d):
        assert (len(x) == self.M)
        self.x = np.concatenate([self.x[self.M:], x])
        X = fft(self.x)
        self.X[1:] = self.X[:-1]
        self.X[0] = X
        Y = np.sum(self.H * self.X, axis=0)
        y = ifft(Y).real[self.M:]
        e = d - y

        e_fft = np.concatenate(
            (np.zeros(shape=(self.M,), dtype=np.float32), e))
        self.E = fft(e_fft)
        self.m = self.m_smooth_factor * self.m + \
                 (1 - self.m_smooth_factor) * np.abs(self.E) ** 2
        R = np.sum(self.X * self.P * self.X.conj(), 0) + 2 * self.m / self.N
        self.mu = self.P / (R + 1e-10)
        if self.res:
            W = 1 - np.sum(self.mu * np.abs(self.X) ** 2, 0)
            E_res = W * self.E
            e = ifft(E_res).real[self.M:].real
            y = d - e
        return e, y

    def update(self):
        G = self.mu * self.X.conj()
        self.P = self.A2 * (1 - 0.5 * G * self.X) * self.P + \
                 (1 - self.A2) * np.abs(self.H) ** 2
        self.H = self.A * (self.H + fft(self.half_window * (ifft(self.E * G).real)))


def pfdkf(x, d, N=10, M=256, A=0.999, P_initial=1, keep_m_gate=0.1):
    ft = PFDKF(N, M, A, P_initial, keep_m_gate)
    num_block = min(len(x), len(d)) // M

    e = np.zeros(num_block * M)
    y = np.zeros(num_block * M)
    for n in range(num_block):
        x_n = x[n * M:(n + 1) * M]
        d_n = d[n * M:(n + 1) * M]
        e_n, y_n = ft.filt(x_n, d_n)
        ft.update()
        e[n * M:(n + 1) * M] = e_n
        y[n * M:(n + 1) * M] = y_n
    return e, y


def main(args):
    session2utt = {}

    with open(args.wav_scp, "r") as f:
        for line in f.readlines():
            utt, path = line.strip().split()
            session2utt.setdefault(utt.split("_")[0], []).append(path)

    sr = 16000
    for session, utts in session2utt.items():
        utts.sort()
        os.makedirs(os.path.join(args.save_path, session), exist_ok=True)

        wav1, _ = sf.read(utts[0])
        wav2, _ = sf.read(utts[1])
        wav3, _ = sf.read(utts[2])
        wav4, _ = sf.read(utts[3])

        ref_utts = glob.glob(os.path.join(args.ref_path, session, "DX0[5-6]C01.wav"))
        if len(ref_utts) == 0:
            sf.write(os.path.join(args.save_path, session, "DX01C01.wav"), wav1, sr)
            sf.write(os.path.join(args.save_path, session, "DX02C01.wav"), wav2, sr)
            sf.write(os.path.join(args.save_path, session, "DX03C01.wav"), wav3, sr)
            sf.write(os.path.join(args.save_path, session, "DX04C01.wav"), wav4, sr)
            continue

        ref_utts.sort()

        mic_list = np.stack([wav1, wav2, wav3, wav4])
        ref1, _ = sf.read(ref_utts[0])
        ref2, _ = sf.read(ref_utts[1])
        errors = []
        for i in range(len(mic_list)):
            mic = mic_list[i]
            error1, echo1 = pfdkf(ref1, mic, A=0.999, keep_m_gate=0.5)
            error2, echo2 = pfdkf(ref2, mic, A=0.999, keep_m_gate=0.5)
            echo = (echo1 + echo2) / 2.0
            min_len = min(len(echo), len(mic))
            mic = mic[:min_len]
            echo = echo[:min_len]
            errors.append(mic - echo)

        out = np.stack(errors, axis=0)

        for i in range(4):
            sf.write(os.path.join(args.save_path, session, f"DX0{i + 1}C01.wav"), out[i], sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", type=str, required=True)
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    main(parser.parse_args())
