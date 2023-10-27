import os
import argparse
import numpy as np
import soundfile as sf

from tqdm import tqdm


def rstft(x, nfft, shift, win):
    nframe = int((len(x) - nfft) / shift) + 1
    X = np.zeros((nfft, nframe), dtype=np.float64)
    begin = 0

    for i in range(nframe):
        X[:, i] = x[begin:begin + nfft]
        begin += shift

    Y = np.fft.fft(X * np.tile(np.expand_dims(win, 1), (1, nframe)), axis=0)
    Y = Y[:nfft // 2 + 1, :]

    return Y


def irstft(Y, shift, win=None):
    Y = np.vstack([Y, np.conj(np.flipud(Y[1:-1, :]))])
    nfft, nframe = Y.shape

    syn_win = np.zeros(nfft)
    for i in range(1, nfft // shift + 1):
        syn_win += np.roll(win, i * shift)

    syn_win = 1.0 / syn_win
    syn_win = syn_win[:shift]

    N = (nframe - 1) * shift + nfft
    x = np.zeros(N)
    begin = 0

    for i in range(nframe):
        x[begin:begin + nfft] += np.real(np.fft.ifft(Y[:, i]))
        begin += shift

    x = x * np.tile(syn_win, N // shift)

    return x


def iva(sp_in, fs=16000, eps=1e-6, epoch=30, nfft=512, nshift=256):
    nTime, M = sp_in.shape
    nf = nfft // 2 + 1

    ana_win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(nfft) / (nfft - 1))

    X = []
    for k in range(M):
        X.append(rstft(sp_in[:, k], nfft, nshift, ana_win))
    X = np.stack(X, axis=2)
    X = X.astype(np.complex64)

    W = np.tile(np.eye(M, dtype=np.complex64).reshape(M, M, 1), (1, 1, nf))
    Winv = np.tile(np.eye(M, dtype=np.complex64).reshape(M, M, 1), (1, 1, nf))
    Wt = np.tile(np.eye(M, dtype=np.complex64).reshape(M, M, 1), (1, 1, nf))
    Vk = np.tile(np.eye(M, dtype=np.complex64).reshape(M, M, 1, 1), (1, 1, nf, M))
    Rx = np.multiply(np.transpose(np.expand_dims(X, 3), (2, 3, 0, 1)),
                     np.transpose(np.conj(np.expand_dims(X, 3)), (3, 2, 0, 1)))

    for iter in tqdm(range(epoch), desc="IVA Iteration"):
        Yp = np.sum(np.multiply(np.transpose(np.expand_dims(W, 3), (0, 1, 3, 2)),
                                np.transpose(np.expand_dims(X, 3), (3, 2, 1, 0))), axis=1)
        R = np.sum(np.real(Yp * np.conj(Yp)), axis=2)
        Gr = 1 / (np.sqrt(R) + eps)

        for k in range(M):
            Vk[:, :, :, k] = np.mean(np.multiply(Rx, Gr[k, :].reshape(1, 1, 1, Gr.shape[1])), axis=3)

            for i in range(nf):
                wk = np.linalg.solve(Vk[:, :, i, k] + 0 * np.eye(M, dtype=np.complex64), Winv[:, k, i])
                wk = wk / (np.sqrt(np.real(np.dot(wk.T.conj(), Winv[:, k, i]))))
                W[k, :, i] = wk.T.conj()

        for i in range(nf):
            Winv[:, :, i] = np.linalg.pinv(W[:, :, i])

    # Normalize W
    for i in range(nf):
        W[:, :, i] = np.dot(np.diag(np.diag(Winv[:, :, i])), W[:, :, i])

    # Get output
    Xp = np.transpose(X, (2, 1, 0))
    Y = Xp
    for i in range(nf):
        Y[:, :, i] = np.dot(W[:, :, i], Xp[:, :, i])

    # iSTFT
    sp_out = []
    for k in range(M):
        sp_out.append(irstft(Y[k, :, :].T, nshift, ana_win))
    sp_out = np.stack(sp_out, axis=0)

    return sp_out


def main(args):
    session2utt = {}

    with open(args.wav_scp, "r") as f:
        for line in f.readlines():
            utt, path = line.strip().split()
            session2utt.setdefault(utt.split("_")[0], []).append(path)

    sr = 16000
    for session, utts in session2utt.items():
        utts.sort()

        wav1, _ = sf.read(utts[0])
        wav2, _ = sf.read(utts[1])
        wav3, _ = sf.read(utts[2])
        wav4, _ = sf.read(utts[3])

        min_length = min(wav1.shape[0], wav2.shape[0], wav3.shape[0], wav4.shape[0])
        x = np.stack([wav1[:min_length], wav2[:min_length], wav3[:min_length], wav4[:min_length]], axis=1)

        y = iva(x)

        os.makedirs(os.path.join(args.save_path, session), exist_ok=True)
        for i in range(4):
            sf.write(os.path.join(args.save_path, session, f"DX0{i + 1}C01.wav"), y[i], sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    main(parser.parse_args())
