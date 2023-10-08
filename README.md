# GNN_SC_FC

Data and codes for our paper **""**.

connectome matrices of  *group-averaged structural, functional, predicted functional connectivity* matrices,  for the **HCP-D** and **HCP-YA** (typically referred as **'HCP'**) datasets using `Schaefer-400` (7 Networks order).  See [data](data/) for more details.

## `data`
- The [sub_info](data/info/sub_info) folder contains the subject information (`sub_id`,`age` and `gender`) used in this study.
- The [info](data/info) folder contains the SC_mask, SA rank
- The [SC_FC_PredFC_matrix](data/SC_FC_PredFC_matrix) folder contains *group-averaged structural, functional, predicted functional connectivity* matrices.
- The [preprocessed_data](data/preprocessed_data) folder contains the preprocessed SC, FC data (please put your SC, FC here in pickle)
- The [result_out_1](data/result_out_1) folder contains the GNN predicted results (please put your PredFC here in pickle)
- The [result_out_2](data/result_out_2) folder contains the group and individual effect 

## `functions`
The [functions](functions/) folder contains code and files commonly used in `code`.

## `result_plot`

The [result_plot](result_plot/) folder contains all figures in paper.

## `code`

- The [step01_ train_val_test_GNN](step01_ train_val_test_GNN/) folder contains our proposed GNN model, and the process of training, validating, and testing the model

  Run the GNN model by

  ```bash
  python train_val_test.py  --epochs 400 --batch-size 2 --lr 0.001 --layer-num 2 --conv-dim 128 --rewired 0 --reg 0.0001  --dataset HCPYA
  ```

  Train and validate the GNN model for hyperparameter tuning using `--if-kfold` for 5 fold cross-validation

  Take rewired SC for training by  `--use-rewired` 

  Add `--get-result` to save the predicted result in `data/result_out` folder

- The [step02_ get_SC_FC_coupling](step02_ get_SC_FC_coupling/) folder contains codes to generate results and figures of *Fig. 2. Structure-function coupling by GNN* and  *Fig. S1* .

  Run `get_SC_mask.m` to get SC masks. 

  Run `Fig2.ipynb` to get structure-function coupling and plot the Fig2. 

  Run `SFig1.ipynb` to get plot the SFig1. 

- The [step03_get_group_ind_effect](step03_get_group_ind_effect/) folder contains codes to generate results and figures of *Fig. 3. group and individual effect of structure-function coupling*. 

  Run `get_group_ind_SC_FC.py` to get group and individual effect of SC-FC. 

  ```bash
  python get_group_ind_SC_FC.py  --dataset HCPA --mask-type 75 # mask-type can be 25 or 75
  ```

  Run `get_group_ind_PredFC_FC.py` to get group and individual effect of PredFC-FC. 

  ```bash
  python get_group_ind_PredFC_FC.py  --dataset HCPA
  ```

  *Note: it highly recommended to run these two scripts in the server. Though I wrote them in parallel, it still cost much time.*

  Run `Fig3.ipynb` to plot the Fig3. 

- The [step04_get_regioan_group_ind_effect](step04_get_regioan_group_ind_effect/) folder contains codes to generate results and figures of *Fig. 4&Fig5. Regional group and individul effect*.

  Run `get_group_ind_SC_FC_roi.py` to get regional group and individual effect of SC-FC. 

  ```bash
  python get_group_ind_SC_FC_roi.py  --dataset HCPA --mask-type 75 # mask-type can be 25 or 75
  ```

  Run `get_group_ind_PredFC_FC_roi.py` to get regional group and individual effect of PredFC-FC. 

  ```bash
  python get_group_ind_PredFC_FC_roi.py  --dataset HCPA
  ```

  *Note: it highly recommended to run these two scripts in the server. Though I wrote them in parallel, it still cost much time.*

  Run `Fig4&5.ipynb` to get the data needed for Fig4&5. 

  Run `get_Fig4&5_nii_from_mat.m` to nil plot for Fig4(a)-(c)&5(a)-(c). 

  Run `Fig4-2.ipynb` and  `Fig5-2.ipynb` to get the data needed for Fig4(d)-(f)&5(d)-(f). 
