clc;
clear;

corr_type = 'pred'; % or sc

load(['/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/result_plot/temp_data', corr_type, '_roi_gp_ind.mat']);

ResultFolder = '/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/result_plot';
%cifti template

template = cifti_read('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/plot/template/Schaefer2018_400Parcels_7Networks_order.dlabel.nii');
template_scalar = cifti_read('/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/plot/template/Schaefer2018_400Parcels_7Networks_order.dscalar.nii');
template = template.cdata;

for roi_i = 1:400
    idx = find(template == roi_i);

    HCPA_roi_gp_o(idx, :) = HCPA_roi_gp(1, roi_i);
    HCPD_roi_gp_o(idx, :) = HCPD_roi_gp(1, roi_i);
    ABCD_roi_gp_o(idx, :) = ABCD_roi_gp(1, roi_i);

    HCPA_roi_ind_o(idx, :) = HCPA_roi_ind(1, roi_i);
    HCPD_roi_ind_o(idx, :) = HCPD_roi_ind(1, roi_i);
    ABCD_roi_ind_o(idx, :) = ABCD_roi_ind(1, roi_i);


end

cifti_energyIn = template_scalar;


cifti_energyIn = template_scalar;
cifti_energyIn.cdata = HCPA_roi_ind_o;
cifti_energyIn.diminfo{2} = cifti_diminfo_make_scalars(size(cifti_energyIn,2));
cifti_write(cifti_energyIn, [ResultFolder, '/Fig5/', 'HCPA_roi_ind_', corr_type, 'dscalar.nii']);


cifti_energyIn = template_scalar;
cifti_energyIn.cdata = HCPD_roi_ind_o;
cifti_energyIn.diminfo{2} = cifti_diminfo_make_scalars(size(cifti_energyIn,2));
cifti_write(cifti_energyIn, [ResultFolder, '/Fig5/', 'HCPD_roi_ind_', corr_type, 'dscalar.nii']);

cifti_energyIn = template_scalar;
cifti_energyIn.cdata = ABCD_roi_ind_o;
cifti_energyIn.diminfo{2} = cifti_diminfo_make_scalars(size(cifti_energyIn,2));
cifti_write(cifti_energyIn, [ResultFolder, '/Fig5/', 'ABCD_roi_ind_', corr_type, '.dscalar.nii']);



cifti_energyIn = template_scalar;
cifti_energyIn.cdata = HCPA_roi_gp_o;
cifti_energyIn.diminfo{2} = cifti_diminfo_make_scalars(size(cifti_energyIn,2));
cifti_write(cifti_energyIn, [ResultFolder, '/Fig4/', 'HCPA_roi_gp_', corr_type, '.dscalar.nii']);

cifti_energyIn = template_scalar;
cifti_energyIn.cdata = HCPD_roi_gp_o;
cifti_energyIn.diminfo{2} = cifti_diminfo_make_scalars(size(cifti_energyIn,2));
cifti_write(cifti_energyIn, [ResultFolder, '/Fig4/', 'HCPD_roi_gp_', corr_type, '.dscalar.nii']);

cifti_energyIn = template_scalar;
cifti_energyIn.cdata = ABCD_roi_gp_o;
cifti_energyIn.diminfo{2} = cifti_diminfo_make_scalars(size(cifti_energyIn,2));
cifti_write(cifti_energyIn, [ResultFolder, '/Fig4/', 'ABCD_roi_gp_', corr_type , '.dscalar.nii']);