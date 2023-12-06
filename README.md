<h1 align="center">ICASSP2024 ICMC-ASR Challenge Baseline</h1>

## Introduction

This repository is the baseline code for the ICMC-ASR (In-Car Multi-Channel Automatic Speech Recognition).

The code in this repository is based on the End-to-End Speech Recognition Toolkit [WeNet](https://github.com/wenet-e2e/wenet) and the Speaker Diarization toolkit [Pyannote-Audio](https://github.com/pyannote/pyannote-audio).

## Data Preparation

Before running this baseline, you should have downloaded and unzipped the dataset for this challenge, whose folder structure is as follows:

```Shell
ICMC-ASR
└── train
|   ├── 001
|   ├── 002
|   ├── 003
|   |   ├── DA0*.wav (near-field headset audios)
|   |   ├── DA0*.TextGrid (label files including speaker label, timestamps and text content)
|   |   ├── DX0[1-4]C01 (4-channel far-field audios)
|   |   └── DX0[5-6]C01 (2-channel reference signals for AEC only existing when the car stereo is turned on)
|   ├── ...
|   └── 568
├── dev
|   ├── 001
|   ├── 002
|   |   ├── DA0*.wav (near-field headset audios)
|   |   ├── DA0*.TextGrid (label files including speaker label, timestamps and text content)
|   |   ├── DX0[1-4]C01 (4-channel far-field audios)
|   |   └── DX0[5-6]C01 (2-channel reference signals for AEC only existing when the car stereo is turned on)
|   ├── ...
|   ├── 018
├── eval_track1 (waiting to release)
|   ├── 001
|   ├── 002
|   |   ├── DA0*.TextGrid (label files including speaker label, timestamps without text content)
|   |   ├── DX0[1-4]C01 (4-channel far-field audios)
|   |   └── DX0[5-6]C01 (2-channel reference signals for AEC only exist when the car stereo is turned on)
|   ├── ...
|   ├── 018
└── eval_track2 (waiting to release)
    ├── 001
    ├── 002
    |   ├── DX0[1-4]C01 (4-channel far-field audios)
    |   └── DX0[5-6]C01 (2-channel reference signals for AEC only exist when the car stereo is turned on)
    ├── ...
    └── 018
```

### Notice:fire: 

We have released the latest near-field audio and textgrids of the training and development sets and fixed the issue with incorrect speaker seat (near-field audio and textgrid's file name) information that was reported by some teams. We strongly recommend you to get them from the download links in the email and replace **all** near-field audio and textgrids from the previously downloaded training and development sets.

## Environment Setup

```Shell
# create environment
conda create -n icmcasr python=3.9 -y
conda activate icmcasr
# install cudatoolkit and cudnn
conda install cudatoolkit=11.6 cudnn=8.4.1 -y
# install pytorch torchvision and torchaudio
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# install other dependence
pip install -r requirements.txt
```

## Track1 Baseline

The main steps are all in track1_asr/run.sh

**[Stage 0]** Audio Frontend AEC + IVA. Segment long waves into short slices and prepare the data files (wav.scp, text, utt2spk and spk2utt).

**[Stage 1]** Compute CMVN value of training set for audio feature normalization.

**[Stage 2]** Generate the token list (char).

**[Stage 3]** Prepare the data in the WeNet required format.

**[Stage 4]** Train the ASR model following the train config file.

**[Stage 5]** Do model average and decoding.

### CER(%) Results

|   Dataset   |          Training Data           | Attention | Attention Rescoring | CTC Greedy Search | CTC Prefix Beam Search |
| :---------: | :------------------------------: | :-------: | :-----------------: | :---------------: | :--------------------: |
|     Dev     | AEC+IVA Far-field and Near-field |   33.28   |        32.92        |       33.66       |         33.66          |
| Eval_Track1 | AEC+IVA Far-field and Near-field |   26.78   |        26.24        |       26.88       |         26.81          |

## Track2  Baseline

Before running the track2 baseline, please make sure you have run all the stages in track1_asr/run.sh and get the trained ASR model.

The VAD model of the track2 baseline is based on [Pyannote-Audio](https://github.com/pyannote/pyannote-audio). The installation steps are as follows.

1. Create a new conda environment with python3.9+ and torch2.0.0+ by the following steps. Or just modify the torch version of the conda env created at track1 baseline.

   ```shell
   # create environment
   conda create -n icmcasr-pyannote python=3.9 -y
   conda activate icmcasr-pyannote
   # install pytorch torchvision and torchaudio
   conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch
   ```

2. Install [`pyannote.audio`](https://github.com/pyannote/pyannote-audio) with `pip install pyannote.audio`

3. Accept [`pyannote/segmentation`](https://hf.co/pyannote/segmentation) user conditions

4. Create your access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).

The main steps are all in track2_asdr/run.sh

**[Stage 0]** Do VAD for enhanced audio data.

**[Stage 1]** Segment audio data based on VAD results.

**[Stage 2]** Prepare the data in the WeNet required format.

**[Stage 3]** Decode using track1 baseline model.

**[Stage 4]** Generate submission file for track2 leaderboard.

**[Stage 5]** Compute cpCER of the dev set.

### cpCER(%) Results

|   Dataset   |          VAD          |      ASR      | Attention | Attention Rescoring | CTC Greedy Search | CTC Prefix Beam Search |
| :---------: | :-------------------: | :-----------: |:---------:|:-------------------:|:-----------------:|:----------------------:|
|     Dev     | pyannote/segmentation | ebranchformer |   67.17   |        65.90        |       66.32       |         66.48          |
| Eval_Track2 | pyannote/segmentation | ebranchformer |   74.29   |        72.88        |       73.19       |         73.52          |

## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.

## Contact

```
[ICMC-ASR Committee](icmcasr_challenge@aishelldata.com)
```
