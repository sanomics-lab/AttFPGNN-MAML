# AttFPGNN-MAML
Meta learning with Attention based FP-GNN for few-shot molecular property prediction


### fs_mol 

1. **calculate Fingerprints**
    ```
    cd fs_mol/meta_mol_relation && python fp_mixed.py
    ```

2. **train**
    ```
    cd ADKF-IFT && bash ../fs_mol/run_train.sh
    ```

3. **test**
    ```
    cd ADKF-IFT && bash ../fs_mol/run_test.sh
    ```

### MoleculeNet

1. **calculate Fingerprints**
    ```
    cd MoleculeNet/data && python fp_mixed.py
    ```

2. **train**
    ```
    cd ADKF-IFT/MoleculeNet && bash ../../MoleculeNet/run_train.sh
    ```

**Model Checkpoint & precaculated fingerprints**

You can download the precaculated fingerprints and model checkpoint from [Google Drive](!https://drive.google.com/file/d/12yT5euhQkbYFr8gZ0mllNAGXXDF0LVcK/view?usp=sharing). After downloading and extracting, you will obtain files related to fs_mol and MoleculeNet respectively.
