import os
import re
import sys
import glob
import math
import textgrid
import traceback
import multiprocessing

from tqdm import tqdm
from functools import partial
from pydub import AudioSegment
from zhon.hanzi import punctuation
from concurrent.futures import ProcessPoolExecutor


def segment_wav(wav_path):
    dur = round(float(os.popen(f"soxi -D {wav_path}").read().strip()), 3)
    # 001_DA02
    wav_id = '_'.join(wav_path.split('/')[-2:]).replace('.wav', '')
    # only segment near-field wavs of training set
    if wav_id.split('_')[-1].startswith("DA0") and dataset != "train":
        return
    utt2text = []
    if wav_id.split('_')[-1].startswith("DA0"):
        wav = AudioSegment.from_wav(wav_path)
        os.system(f"mkdir -p {wav_path.replace('.wav', '')}")
        tg = textgrid.TextGrid()
        try:
            tg.read(wav_path.replace('.wav', '.TextGrid'))
        except:
            print(f"Error With: {wav_path.replace('.wav', '.TextGrid')}")
            return
        for tier in tg.tiers:
            speaker = tier.name
            for interval in tier:
                if not interval.mark or interval.mark == "":
                    continue
                text = re.sub(f"[{punctuation} *]", "", interval.mark)
                if text == "":
                    continue
                start, end = float(interval.minTime), float(interval.maxTime)
                if end > dur or end - start < 0.03:
                    continue
                # P0117_001_DA02_005156-006106.wav
                utterance_id = f"{'_'.join([speaker] + wav_id.split('_'))}_{int(start * 1000):0>6}-{int(end * 1000):0>6}"
                utt2text.append((utterance_id, text))
                export_wav_path = f"{wav_path.replace('.wav', '')}/{utterance_id}.wav"
                wav[start * 1000:end * 1000].export(export_wav_path, format="wav")
        with open(f"{'/'.join(wav_path.split('/')[:-1])}/{wav_path.split('/')[-1].replace('.wav', '.txt')}", 'w') as f:
            for item in utt2text:
                f.write(f"{item[0]} {item[1]}\n")
    else:
        # for dev and eval_track1 sets
        # segment the enhanced far-field wavs
        enhanced_wav_path = wav_path.replace(data_root, enhanced_data_root)
        wav = AudioSegment.from_wav(enhanced_wav_path)
        tg_files = glob.glob(f"{os.path.dirname(wav_path)}/DA0*.TextGrid")
        for tg_file in tg_files:
            # only use the far-field wav closest to the speaker for training
            if wav_id.split('_')[1][3] != tg_file.split('/')[-1][3]:
                continue
            os.system(f"mkdir -p {enhanced_wav_path.replace('.wav', '')}")
            tg = textgrid.TextGrid()
            try:
                tg.read(tg_file)
            except:
                print(f"Error With: {tg_file}")
                continue
            for tier in tg.tiers:
                speaker = tier.name
                for interval in tier:
                    if not interval.mark or interval.mark == "":
                        continue
                    text = re.sub(f"[{punctuation} *]", "", interval.mark)
                    if text == "":
                        continue
                    start, end = float(interval.minTime), float(interval.maxTime)
                    if end > dur or end - start < 0.03:
                        continue
                    # P0117_001_DA02_005156-006106.wav
                    utterance_id = f"{'_'.join([speaker] + wav_id.split('_'))}_{int(start * 1000):0>6}-{int(end * 1000):0>6}"
                    utt2text.append((utterance_id, text))
                    export_wav_path = f"{enhanced_wav_path.replace('.wav', '')}/{utterance_id}.wav"
                    wav[start * 1000:end * 1000].export(export_wav_path, format="wav")
            with open(f"{'/'.join(wav_path.split('/')[:-1])}/{wav_path.split('/')[-1].replace('.wav', '.txt')}", 'w') as f:
                for item in utt2text:
                    f.write(f"{item[0]} {item[1]}\n")


def multiThread_use_ProcessPoolExecutor_dicarg(scp, numthread, func, args):
    executor = ProcessPoolExecutor(max_workers=numthread)
    results = []
    for item in scp:
        results.append(executor.submit(partial(func, item, **args)))
    return [result.result() for result in tqdm(results)]


if __name__ == '__main__':
    data_root, enhanced_data_root, dataset, nj = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    dataset = dataset.split('_aec_iva')[0]
    audio_root = f"{data_root}/{dataset}"
    wav_scp = [wav for wav in glob.glob(f'{audio_root}/*/*.wav') if not wav.split('/')[-1].startswith(("DX05", "DX06"))]
    wav_scp.sort()
    multiThread_use_ProcessPoolExecutor_dicarg(wav_scp, nj, segment_wav, {})
