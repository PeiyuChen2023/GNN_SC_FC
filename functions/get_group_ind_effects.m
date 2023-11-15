function [group_effect,ind_effect,match_corr,mismatch_corr] = get_group_ind_effects(data_corr)

[sub_num,~] = size(data_corr);
intra_mask = eye(sub_num,sub_num);

% calculate the group and individual effects
group_effect = mean(data_corr(intra_mask == 0));
ind_effect = mean(data_corr(intra_mask == 1)) - group_effect;

% calculate the matched and mismatched correlation for each subject
match_corr = data_corr(1:sub_num+1:end)';

mismatch_corr = zeros(sub_num,1);
for sub_i = 1:sub_num
    mismatch_col = data_corr(sub_i,setdiff(1:sub_num,sub_i));
    mismatch_row = data_corr(setdiff(1:sub_num,sub_i),sub_i);
    mismatch_corr(sub_i,1) = mean([mismatch_col(:);mismatch_row(:)]);
end