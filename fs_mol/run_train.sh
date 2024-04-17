#!/bin/bash

SCRIPT_PATH=$(cd $(dirname $0);pwd)
echo "PATH: $SCRIPT_PATH"

pushd ../ADKF-IFT

python ${SCRIPT_PATH}/meta_mol_relation/meta_mol_relation_train.py ./datasets \
    --features gnn+ecfp+mixed+fc \
    --use_attention \
    --padding_label_dim 64 \
    --num_inner_steps 3

popd
