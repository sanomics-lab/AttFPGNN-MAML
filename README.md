# AttFPGNN-MAML
Meta learning with Attention based FP-GNN for few-shot molecular property prediction


### fs_mol 

1. **calculate Fingerprints**
    ```
    cd fs_mol/meta_mol_relation && python fp_mixed.py
    ```

2. **train**
    ```
    python fs_mol/meta_mol_relation/meta_mol_relation_train.py /path/to/data
    ```

3. **test**
    ```
    python fs_mol/meta_mol_relation/meta_mol_relation_test.py /path/to/pn-checkpoint /path/to/data
    ```

### MoleculeNet

1. **calculate Fingerprints**
    ```
    cd MoleculeNet/data && python fp_mixed
    ```

2. **train**
    ```
    cd MoleculeNet && bash run_train.sh
    ```
