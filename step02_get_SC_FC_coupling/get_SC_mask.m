root_dir = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/';

func_dir = [root_dir 'functions'];
addpath(genpath(func_dir))
SC_path = [root_dir 'data/preprocessed_data/All_SC.mat'];
load(SC_path)
W_thr = threshold_consistency(HCPA_SC, 0.25);
HCPA_sc_mask = double(W_thr > 0);
W_thr = threshold_consistency(HCPD_SC, 0.25);
HCPD_sc_mask = double(W_thr > 0);
W_thr = threshold_consistency(ABCD_SC, 0.25);
ABCD_sc_mask = double(W_thr > 0);
save([root_dir 'data/info/SC_mask_25.mat'], "HCPA_sc_mask", "HCPD_sc_mask", "ABCD_sc_mask"); 
W_thr = threshold_consistency(HCPA_SC, 0.75);
HCPA_sc_mask = double(W_thr > 0);
W_thr = threshold_consistency(HCPD_SC, 0.75);
HCPD_sc_mask = double(W_thr > 0);
W_thr = threshold_consistency(ABCD_SC, 0.75);
ABCD_sc_mask = double(W_thr > 0);
save([root_dir 'data/info/SC_mask_75.mat'], "HCPA_sc_mask", "HCPD_sc_mask", "ABCD_sc_mask"); 