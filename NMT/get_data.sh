# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/bin/bash
set -e

#
# Data preprocessing configuration
#

CODES=160  # Vocabulary size

#
# Initialize tools and data paths
#

# main paths
SRC=asp
TGT=csj
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
SRC_PATH=$PWD/src
DATA_PATH=$PWD/data/test/${SRC}_${TGT}

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast

# files full paths
SRC_TRN=$DATA_PATH/${SRC}.train
TGT_TRN=$DATA_PATH/${TGT}.train
SRC_VAL=$DATA_PATH/${SRC}.valid
TGT_VAL=$DATA_PATH/${TGT}.valid
SRC_TEST=$DATA_PATH/${SRC}.test
TGT_TEST=$DATA_PATH/${TGT}.test
BPE_CODES=$DATA_PATH/bpe_codes
SRC_VOCAB=$DATA_PATH/vocab.${SRC}.$CODES
TGT_VOCAB=$DATA_PATH/vocab.${TGT}.$CODES
FULL_VOCAB=$DATA_PATH/vocab.all.$CODES

# Download fastBPE
mkdir -p $TOOLS_PATH
cd $TOOLS_PATH
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi
echo "fastBPE found in: $FASTBPE_DIR"

# Compile fastBPE
cd $TOOLS_PATH
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd $FASTBPE_DIR
  g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
fi
echo "fastBPE compiled in: $FASTBPE"

# apply MeCab Tokenization
for RAW_DATA in $SRC_TRN $TGT_TRN $SRC_VAL $TGT_VAL $SRC_TEST $TGT_TEST
do
  python $SRC_PATH/modules/mecab_tokenizer.py -src $RAW_DATA -out $RAW_DATA.tok
done

# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $SRC_TRN.tok $TGT_TRN.tok > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"

# apply BPE codes for train data
if ! [[ -f "$SRC_TRN.$CODES" && -f "$TGT_TRN.$CODES" ]]; then
  echo "Applying BPE codes..."
  $FASTBPE applybpe $SRC_TRN.$CODES $SRC_TRN.tok $BPE_CODES
  $FASTBPE applybpe $TGT_TRN.$CODES $TGT_TRN.tok $BPE_CODES
fi
echo "BPE codes applied to SRC in: $SRC_TRN.$CODES"
echo "BPE codes applied to TGT in: $TGT_TRN.$CODES"

# extract vocabulary
if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" && -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TRN.$CODES > $SRC_VOCAB
  $FASTBPE getvocab $TGT_TRN.$CODES > $TGT_VOCAB
  $FASTBPE getvocab $SRC_TRN.$CODES $TGT_TRN.$CODES > $FULL_VOCAB
fi
echo "SRC vocab in: $SRC_VOCAB"
echo "TGT vocab in: $TGT_VOCAB"
echo "Full vocab in: $FULL_VOCAB"

# binarize data
if ! [[ -f "$SRC_TRN.$CODES.pth" && -f "$TGT_TRN.$CODES.pth" ]]; then
  echo "Binarizing data..."
  python $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TRN.$CODES
  python $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TRN.$CODES
fi
echo "SRC binarized data in: $SRC_TRN.$CODES.pth"
echo "TGT binarized data in: $TGT_TRN.$CODES.pth"

# apply BPE codes for valid/test data
echo "Applying BPE codes to valid/test..."
$FASTBPE applybpe $SRC_VAL.$CODES $SRC_VAL.tok $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_VAL.$CODES $TGT_VAL.tok $BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $SRC_TEST.$CODES $SRC_TEST.tok $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_TEST.$CODES $TGT_TEST.tok $BPE_CODES $TGT_VOCAB

# binarize valid/test data
echo "Binarizing valid/test data..."
rm -f $SRC_VAL.$CODES.pth $TGT_VAL.$CODES.pth $SRC_TEST.$CODES.pth $TGT_TEST.$CODES.pth
python $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VAL.$CODES
python $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VAL.$CODES
python $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST.$CODES
python $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST.$CODES

#
# Summary
#
echo ""
echo "===== Data summary"
echo "Monolingual training data:"
echo "    SRC: $SRC_TRN.$CODES.pth"
echo "    TGT: $TGT_TRN.$CODES.pth"
echo "Monolingual validation data:"
echo "    SRC: $SRC_VAL.$CODES.pth"
echo "    TGT: $TGT_VAL.$CODES.pth"
echo "Monolingual test data:"
echo "    SRC: $SRC_TEST.$CODES.pth"
echo "    TGT: $TGT_TEST.$CODES.pth"
echo ""
