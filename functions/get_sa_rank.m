clear
clc

cd('E:\OtherGroup\Ted\S-A_ArchetypalAxis\FSLRVertex')

%%
sa_rank = cifti_read('SensorimotorAssociation_Axis.dscalar.nii');
sa_rank_left = cifti_struct_dense_extract_surface_data(sa_rank,'CORTEX_LEFT');
sa_rank_right = cifti_struct_dense_extract_surface_data(sa_rank,'CORTEX_RIGHT');
sa_rank_all = [sa_rank_left;sa_rank_right];

%%
schaefer100 = cifti_read("E:\toolbox\CBIG\stable_projects\brain_parcellation\Schaefer2018_LocalGlobal\Parcellations\HCP\fslr32k\cifti\Schaefer2018_100Parcels_7Networks_order.dlabel.nii");
schaefer100 = schaefer100.cdata;

for roi_i = 1:100
    idx = find(schaefer100 == roi_i);
    sa_rank_schaefer100(roi_i,1) = mean(sa_rank_all(idx,:));
end
[~,sa_rank_schaefer100] = sort(sa_rank_schaefer100,'ascend');
[~,sa_rank_schaefer100] = sort(sa_rank_schaefer100,'ascend');

save('F:\Cui_Lab\Projects\GNN_SC_FC\matlab\data\sa_rank_schaefer100.mat','sa_rank_schaefer100')

%%
schaefer200 = cifti_read("E:\toolbox\CBIG\stable_projects\brain_parcellation\Schaefer2018_LocalGlobal\Parcellations\HCP\fslr32k\cifti\Schaefer2018_200Parcels_7Networks_order.dlabel.nii");
schaefer200 = schaefer200.cdata;

for roi_i = 1:200
    idx = find(schaefer200 == roi_i);
    sa_rank_schaefer200(roi_i,1) = mean(sa_rank_all(idx,:));
end
[~,sa_rank_schaefer200] = sort(sa_rank_schaefer200,'ascend');
[~,sa_rank_schaefer200] = sort(sa_rank_schaefer200,'ascend');

save('F:\Cui_Lab\Projects\GNN_SC_FC\matlab\data\sa_rank_schaefer200.mat','sa_rank_schaefer200')

%%
brainnetome = cifti_read("F:\Cui_Lab\Projects\backup_p01\backup9\Connectional_Variability_Gradient\data\parcellation_files\BN_Atlas.dlabel.nii");
brainnetome = brainnetome.cdata;

for roi_i = 1:210
    idx = find(brainnetome == roi_i);
    sa_rank_brainnetome(roi_i,1) = mean(sa_rank_all(idx,:));
end
[~,sa_rank_brainnetome] = sort(sa_rank_brainnetome,'ascend');

save('F:\Cui_Lab\Projects\GNN_SC_FC\matlab\data\sa_rank_brainnetome.mat','sa_rank_brainnetome')


