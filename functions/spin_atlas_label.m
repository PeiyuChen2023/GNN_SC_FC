clear
clc
addpath(genpath('.../ENIGMA/matlab'))

%% schaefer_400
lsphere = SurfStatReadSurf1('fsa5_sphere_lh');
rsphere = SurfStatReadSurf1('fsa5_sphere_rh');

lh_centroid = centroid_extraction_sphere(lsphere.coord.', 'fsa5_lh_schaefer_400.annot');
rh_centroid = centroid_extraction_sphere(rsphere.coord.', 'fsa5_rh_schaefer_400.annot');

perm_id = rotate_parcellation(lh_centroid, rh_centroid, 10000);

save('F:/Cui_Lab/Projects/GNN_SC_FC/matlab/data/perm_id_schaefer400.mat','perm_id')

%% schaefer_200
lsphere = SurfStatReadSurf1('fsa5_sphere_lh');
rsphere = SurfStatReadSurf1('fsa5_sphere_rh');

lh_centroid = centroid_extraction_sphere(lsphere.coord.', 'fsa5_lh_schaefer_200.annot');
rh_centroid = centroid_extraction_sphere(rsphere.coord.', 'fsa5_rh_schaefer_200.annot');

perm_id = rotate_parcellation(lh_centroid, rh_centroid, 10000);

save('F:/Cui_Lab/Projects/GNN_SC_FC/matlab/data/perm_id_schaefer200.mat','perm_id')

%% schaefer_100
lsphere = SurfStatReadSurf1('fsa5_sphere_lh');
rsphere = SurfStatReadSurf1('fsa5_sphere_rh');

lh_centroid = centroid_extraction_sphere(lsphere.coord.', 'fsa5_lh_schaefer_100.annot');
rh_centroid = centroid_extraction_sphere(rsphere.coord.', 'fsa5_rh_schaefer_100.annot');

perm_id = rotate_parcellation(lh_centroid, rh_centroid, 10000);

save('F:/Cui_Lab/Projects/GNN_SC_FC/matlab/data/perm_id_schaefer100.mat','perm_id')
