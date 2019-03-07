df_all
frac_CV = 0.2
df_tr, df_cv = sk.model_selection.train_test_split(df_tr, test_size=frac_CV)