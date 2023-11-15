function [r,p_spin,r_spin] = corr_sa_spin(data,sa_rank,perm_id,rand_num)

non_outlier = abs(zscore(data)) <= 3;
r = corr(data(non_outlier),sa_rank(non_outlier),"type","Spearman");

for i = 1:rand_num
    data_temp = data(perm_id(:,i));
    r_spin(i,1) = corr(data_temp(non_outlier),sa_rank(non_outlier),"type","Spearman");
end

if r > 0
    p_spin = (1+length(find(r_spin >= r)))/(rand_num+1);
else
    p_spin = (1+length(find(r_spin <= r)))/(rand_num+1);
end