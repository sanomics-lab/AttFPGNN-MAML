#!/bin/bash

python -u -m pdb -c c main_mamlmol.py --epochs 2000 --eval_steps 10 --pretrained 0 --n-shot-train 10 --n-shot-test 10 --n-query 32 --dataset muv --test-dataset muv --seed 0 --gpu_id 0 --meta-lr 0.0001
python -u -m pdb -c c main_mamlmol.py --epochs 2000 --eval_steps 10 --pretrained 0 --n-shot-train 10 --n-shot-test 10 --n-query 32 --dataset sider --test-dataset sider --seed 0 --gpu_id 0 --meta-lr 0.0001
python -u -m pdb -c c main_mamlmol.py --epochs 2000 --eval_steps 10 --pretrained 0 --n-shot-train 10 --n-shot-test 10 --n-query 32 --dataset tox21 --test-dataset tox21 --seed 0 --gpu_id 0 --meta-lr 0.0001
python -u -m pdb -c c main_mamlmol.py --epochs 2000 --eval_steps 10 --pretrained 0 --n-shot-train 10 --n-shot-test 10 --n-query 32 --dataset toxcast --test-dataset toxcast --seed 0 --gpu_id 0 --meta-lr 0.0001
