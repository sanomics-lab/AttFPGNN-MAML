#!/bin/bash

SCRIPT_PATH=$(cd $(dirname $0);pwd)
echo "PATH: $SCRIPT_PATH"

CKPT_FILE="/path/to/model.pt"

pushd ../ADKF-IFT

python ${SCRIPT_PATH}/meta_mol_relation/meta_mol_relation_test.py ${CKPT_FILE} ./datasets

popd
