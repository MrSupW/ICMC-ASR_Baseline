import os
import sys
import glob
import argparse
import numpy as np
import soundfile as sf

sys.path.append("./enhancement")
from tqdm import tqdm
from enhancement.iva import iva
from enhancement.pfdkf import pfdkf


def main(args):
    session2utt = {}

    with open(args.wav_scp, "r") as f:
        for line in f.readlines():
            utt, path = line.strip().split()
            session2utt.setdefault(utt.split("_")[0], []).append(path)

    sr = 16000
    for session, utts in tqdm(session2utt.items(), desc="Enhancing"):
        utts.sort()
        os.makedirs(os.path.join(args.save_path, session), exist_ok=True)

        wav1, _ = sf.read(utts[0])
        wav2, _ = sf.read(utts[1])
        wav3, _ = sf.read(utts[2])
        wav4, _ = sf.read(utts[3])

        min_length = min(wav1.shape[0], wav2.shape[0], wav3.shape[0], wav4.shape[0])
        x = np.stack([wav1[:min_length], wav2[:min_length], wav3[:min_length], wav4[:min_length]], axis=1)

        y = iva(x)

        ref_path = os.path.dirname(utts[0])
        ref_utts = glob.glob(os.path.join(ref_path, session, "DX0[5-6]C01.wav"))
        if len(ref_utts) == 0:
            for i in range(4):
                sf.write(os.path.join(args.save_path, session, f"DX0{i + 1}C01.wav"), y[i], sr)
            continue

        ref_utts.sort()
        ref1, _ = sf.read(ref_utts[0])
        ref2, _ = sf.read(ref_utts[1])

        errors = []
        for i in range(len(y)):
            mic = y[i]
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
    parser.add_argument("--save_path", type=str, required=True)
    main(parser.parse_args())
