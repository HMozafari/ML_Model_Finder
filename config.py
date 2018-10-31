


dataset_file_add = "Agrimind_yield_climate_data_total_trimmed_lag_1.xlsx"
train_set_precental = 0.9 # it means that 90% of entire_dataset is dedicated to training set that itself is broke-down into train and validation.

# Feature engineering operations list ['Add', 'Sub', 'Mult', 'Div', 'Sin', 'Cos', 'Exp']
oprs_for_feat_eng = ['Mult', 'Div']

# maximum runtime in secs
max_run_time = 300

num_of_most_corr_feats = 20

# the method to do feature selection
# ['Most-Corr', 'UniVar-Select', 'Rec-Feat-Elimin', 'Prin-Comp-Anal', 'Feat-Import']
feature_select_methods = "Feat-Import"

verbose = 5

# which model to use to do ML models design space search
model_finder = "H2O"

# This flag determines if we want to have stat graphs as files
dataset_stat_visual_flag = False

