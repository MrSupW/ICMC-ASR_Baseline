. ./path.sh || exit 1

stage=0
stop_stage=0
nj=48

. tools/parse_options.sh

data_root=$1
enhanced_data_root=$2
dataset=$3

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  dataset_pf=$(echo "$dataset" | cut -d '_' -f 1)
  if [ ${dataset_pf} == 'eval' ]; then
    dataset_pf=$(echo "$dataset" | cut -d _ -f 1-2)
  fi

  echo "[local/enhancement.sh] generate enhanced audio for ${dataset_pf}"

  mkdir -p exp/enhance/${dataset_pf}

  ls ${data_root}/${dataset_pf}/*/DX0[1-4]C01.wav | awk -F/ '{print $(NF-1)"_"substr($NF, 1, length($NF)-4), $0}' \
    > exp/enhance/${dataset_pf}/wav.scp

  mkdir -p exp/enhance/${dataset_pf}/split_scp
  mkdir -p exp/enhance/${dataset_pf}/log

  file_len=`wc -l exp/enhance/${dataset_pf}/wav.scp | awk '{print $1}'`
  subfile_len=$[${file_len} / ${nj} + 1]

  if [ $[${file_len} / 4] -le ${nj} ]; then
    nj=$[${file_len} / 4]
    subfile_len=4
  fi

  prefix='split'
  split -l $subfile_len -d -a 3 exp/enhance/${dataset_pf}/wav.scp exp/enhance/${dataset_pf}/split_scp/${prefix}_scp_
  echo "you can check the log files in exp/enhance/${dataset_pf}/log for possible errors and progress"
  for suffix in `seq 0 $[${nj}-1]`;do
      suffix=`printf '%03d' $suffix`
      scp_subfile=exp/enhance/${dataset_pf}/split_scp/${prefix}_scp_${suffix}
      python3 -u local/enhance.py \
          --wav_scp ${scp_subfile} \
          --save_path ${enhanced_data_root}/${dataset_pf} \
          > exp/enhance/${dataset_pf}/log/${prefix}.${suffix}.log 2>&1 &
  done
  wait
fi
