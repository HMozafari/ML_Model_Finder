


dataset_file_add = "Agrimind_yield_climate_data_total_trimmed_lag_1.xlsx"
train_set_precental = 0.9 # it means that 90% of entire_dataset is dedicated to training set that itself is broke-down into train and validation.

# Feature engineering operations list
oprs_for_feat_eng = ['Add', 'Sub', 'Mult', 'Div', 'Sin', 'Cos', 'Exp']

# maximum runtime in secs
max_run_time = 3000

num_of_most_corr_feats = 20

feature_select_methods = "Feat-Import"

verbose = 5

model_finder = "H2O"

