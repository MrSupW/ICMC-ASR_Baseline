import os
import re
import sys
import glob
import math
import argparse
import textgrid
import traceback
import multiprocessing

from tqdm import tqdm
from functools import partial
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor


def rttm2wav(line):
    line = line.strip()
    _, session, _, start, duration, _, _, seat, _, _ = line.split()
    wav = AudioSegment.from_wav(f"{enhanced_data_root}/{dataset.split('_aec_iva')[0]}/{session}/DX0{seat}C01.wav")
    start = int(float(start) * 1000)
    end = start + int(float(duration) * 1000)
    export_wav_path = f"{output_dir}/P000{seat}_{session}_DX0{seat}C01_{start:0>6}-{end:0>6}.wav"
    wav[start:end].export(export_wav_path, format="wav")


def multiThread_use_ProcessPoolExecutor_dicarg(scp, numthread, func, args):
    executor = ProcessPoolExecutor(max_workers=numthread)
    results = []
    for item in scp:
        results.append(executor.submit(partial(func, item, **args)))
    return [result.result() for result in tqdm(results)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enhanced_data_root', type=str, help='enhanced data root')
    parser.add_argument('--dataset', type=str, help='dataset')
    parser.add_argument('--nj', type=int, default=32, help='number of jobs')
    parser.add_argument('rttm_file', help='rttm file path')
    parser.add_argument('output_dir', help='output dir of segment wavs')
    args = parser.parse_args()
    enhanced_data_root, dataset, nj, rttm_file, output_dir = \
        args.enhanced_data_root, args.dataset, args.nj, args.rttm_file, args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    lines = open(rttm_file, 'r').readlines()
    # segment_wavs_by_rttm
    multiThread_use_ProcessPoolExecutor_dicarg(lines, nj, rttm2wav, {})
    # write scp and blank text
    output_dir_dir = '/'.join(output_dir.strip('/').split('/')[:-1])
    with open(f"{output_dir_dir}/wav.scp", 'w') as ws:
        with open(f"{output_dir_dir}/text", 'w') as tt:
            for wav in glob.glob(f"{output_dir}/*.wav"):
                ws.write(f"{os.path.basename(wav).split('.')[0]} {wav}\n")
                tt.write(f"{os.path.basename(wav).split('.')[0]} 空\n")
