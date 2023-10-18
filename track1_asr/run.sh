#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# for debug purpose, please set it to 1 otherwise, set it to 0
export CUDA_LAUNCH_BLOCKING=0

stage=0 # start from 0 if you need to start from data preparation
stop_stage=0
data_prep_stage=0  # stage for data preparation
data_prep_stop_stage=2

################################################
# The icmc-asr dataset location, please change this to your own path!!!
# make sure of using absolute path. DO-NOT-USE relatvie path!
data=/home/work_nfs7/hwang/data/ICMC-ASR
# data dir for IVA + AEC enhanced audio
data_enhanced=/home/work_nfs7/hwang/data/ICMC-ASR_ENHANCED
################################################

nj=48
dict=data/dict/lang_char.txt

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

train_set=train_iva_aec_near
dev_set=dev_iva_aec
test_set=dev_iva_aec
train_config=conf/train_ebranchformer.yaml
cmvn=true
dir=exp/baseline_ebranchformer
checkpoint=
num_workers=8
prefetch=500
find_unused_parameters=false

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=30
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Frontend IVA + AEC + Data preparation
  echo "stage 0: Frontend IVA + AEC + Data preparation"
  for x in ${dev_set} ${train_set}; do
    local/icmcasr_data_prep.sh --stage ${data_prep_stage} --stop_stage ${data_prep_stop_stage} \
      --nj ${nj} ${data} ${data_enhanced} ${x}
  done
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # compute cmvn
  echo "stage 1: Compute cmvn"
  tools/compute_cmvn_stats.py --num_workers ${nj} --train_config $train_config \
    --in_scp data/${train_set}/wav.scp \
    --out_cmvn data/${train_set}/global_cmvn
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "stage 2: Make a dictionary"
  mkdir -p $(dirname $dict)
  echo "<blank> 0" > ${dict}  # 0 is for "blank" in CTC
  echo "<unk> 1"  >> ${dict}  # <unk> must be 1
  tools/text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " \
    | tr " " "\n" | sort | uniq | grep -a -v -e '^\s*$' | \
    awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Prepare data, prepare required format"
  for x in ${dev_set} ${train_set}; do
    if [ $data_type == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads ${nj} data/$x/wav.scp data/$x/text \
        $(realpath data/$x/shards) data/$x/data.list
    else
      tools/make_raw_list.py data/$x/wav.scp data/$x/text \
        data/$x/data.list
    fi
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p $dir
  INIT_FILE=$dir/ddp_init
  rm -f $INIT_FILE # delete old one before starting
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  echo "num_gpus: $num_gpus"
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  cmvn_opts=
  $cmvn && cmvn_opts="--cmvn data/${train_set}/global_cmvn"

  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    echo "GPU $i: train gpu_id $gpu_id"
    python3 wenet/bin/train.py --gpu "$gpu_id" \
      --config $train_config \
      --data_type $data_type \
      --symbol_table $dict \
      --train_data data/$train_set/data.list \
      --cv_data data/$dev_set/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method "$init_method" \
      --ddp.world_size "$num_gpus" \
      --ddp.rank $i \
      --ddp.dist_backend $dist_backend \
      --num_workers $num_workers \
      --prefetch $prefetch \
      --find_unused_parameters $find_unused_parameters \
      $cmvn_opts \
      --pin_memory
  } &
  done
  wait
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python3 wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decoding_chunk_size=
  ctc_weight=0.3
  idx=0
  for mode in ${decode_modes}; do
  {
    test_dir=$dir/${test_set}_${mode}
    mkdir -p $test_dir
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
    python3 wenet/bin/recognize.py --gpu $gpu_id \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type $data_type \
      --test_data data/${test_set}/data.list \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --result_file $test_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    python3 tools/compute-wer.py --char=1 --v=1 \
      data/${test_set}/text $test_dir/text > $test_dir/wer
  } &
  ((idx+=1))
  if [ $idx -eq $num_gpus ]; then
    idx=0
  fi
  done
  wait
fi
