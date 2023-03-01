#!/usr/bin/env bash

# Copyright 2019 QCRI (Author:Ahmed Ali)
# Apache 2.0

# Copyright 2023 NAME REDACTED FOR ANONYMITY IN THE REVIEW PROCESS
# Apache 2.0

# Supplementary material to the paper "Whistle-to-text: Automatic recognition of the Silbo Gomero whistled language"
# This is the Kaldi recipe that I used to train both of the ASR models described in the paper.

name=
stage=0

# initialization PATH
. ./path.sh  || die "path.sh expected";
# initialization commands
. ./cmd.sh
. ./utils/parse_options.sh

set -e -o pipefail


nj=4
dev_nj=4


if [ $stage -le 1 ]; then
  # Feature extraction
  for x in train dev; do
      steps/make_mfcc_pitch.sh --nj $nj --cmd "$train_cmd" data/$x exp/make_mfcc/$x mfcc
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
  done
fi

if [ $stage -le 2 ]; then
  ### Monophone
  echo "Starting monophone training."
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono
  echo "Mono training done."


  echo "Decoding the dev set using monophone models."
  utils/mkgraph.sh data/lang exp/mono exp/mono/graph

  steps/decode.sh --config conf/decode.config --nj $dev_nj --cmd "$train_cmd" \
    exp/mono/graph data/dev exp/mono/decode_dev
  echo "Monophone decoding done."

fi

if [ $stage -le 3 ]; then
  ### Triphone
  echo "Starting triphone training."
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/mono exp/mono_ali
  steps/train_deltas.sh --cmd "$train_cmd"  \
      4200 40000 data/train data/lang exp/mono_ali exp/tri1
  echo "Triphone training done."

  echo "Decoding the dev set using triphone models."
  utils/mkgraph.sh data/lang  exp/tri1 exp/tri1/graph
  steps/decode.sh --nj $dev_nj --cmd "$decode_cmd"  \
      exp/tri1/graph  data/dev exp/tri1/decode_dev

  echo "Triphone decoding done."
fi



if [ $stage -le 4 ]; then
  ### Triphone + LDA and MLLT
  # Training
  echo "Starting LDA+MLLT training."
  steps/align_si.sh --nj $nj --cmd "$train_cmd"  \
      data/train data/lang exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd"  \
    --splice-opts "--left-context=3 --right-context=3" \
    4200 40000 data/train data/lang exp/tri1_ali exp/tri2b
  echo "LDA+MLLT training done."

  echo "Decoding the dev set using LDA+MLLT models."
  utils/mkgraph.sh data/lang exp/tri2b exp/tri2b/graph
  steps/decode.sh --nj $dev_nj --cmd "$decode_cmd" \
      exp/tri2b/graph data/dev exp/tri2b/decode_dev

  echo "LDA+MLLT decoding done."
fi


if [ $stage -le 5 ]; then
  ### Triphone + LDA and MLLT + SAT and FMLLR
  # Training
  echo "Starting SAT+FMLLR training."
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali
  steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
      data/train data/lang exp/tri2b_ali exp/tri3b
  echo "SAT+FMLLR training done."

  echo "Decoding the dev set using SAT+FMLLR models."
  utils/mkgraph.sh data/lang  exp/tri3b exp/tri3b/graph
  steps/decode_fmllr.sh --nj $dev_nj --cmd "$decode_cmd" \
      exp/tri3b/graph  data/dev exp/tri3b/decode_dev

  echo "SAT+FMLLR decoding done."

fi



wait;

#score
for x in exp/chain/*/decode* exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done | sort -k2 -n > RESULTS_WER_$name

for x in exp/chain/*/decode* exp/*/decode*; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done | sort -k2 -n > RESULTS_CER_$name
