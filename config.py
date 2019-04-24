

# gem AV_TUIN1:Afd 4_g/m3_night_max univ. meting_TUIN2:ALGEMEEN 1 NO_Unnamed: 1_level_2_night_accumulation
dataset_file_add = "../AgriMind_Yield_Model_Development/thanos_output_agrimind_wiht_feat_20190415103917.csv" #"../Rimato/rimato1_weekly_total_df_with_feat20190411131814.csv" #"../Rimato/rimato_test.csv" #   #"../Rimato/rimato_test.csv" #"../SERA/SERA_without_Features_Hasan_removed_rows_without_actual_yield.csv" # ../SERA/thanos_output_new_sera_with_features.csv"#../AgriMind_Yield_Model_Development/thanos_output_agrimind_NIC.csv" #"../Rimato/Adama_rimato_NIC_y6_orig.csv" #"./thanos_output_agrimind_NIC.csv"   # #"../Rimato/thanos_output_rimato_Scott.csv" #"./Test_dataset.csv" #"../SERA/thanos_output_sera.csv"  #"Removed_Rows_Without_Yield.csv" #"thanos_output_agrimind.csv"   #"Agrimind_yield_climate_data_total_trimmed_lag_1.xlsx"
train_set_precental = 0.75 # it means that 90% of entire_dataset is dedicated to training set that itself is broke-down into train and validation.

# Feature engineering operations list ['Add', 'Sub', 'Mult', 'Div', 'Sin', 'Cos', 'Exp']
oprs_for_feat_eng = """["Add", "Sub"]"""

# maximum runtime in secs
max_run_time = 30

num_of_most_corr_feats = 2

# the method to do feature selection
# ['Most-Corr', 'UniVar-Select', 'Rec-Feat-Elimin', 'Prin-Comp-Anal', 'Feat-Import']
feature_select_methods = "Most-Corr" #"Feat-Import"

verbose = 5

# which model to use to do ML models design space search
model_finder = "H2O"
# This flag determines if we want to have stat graphs as files
dataset_stat_visual_flag = False

rmv_feat_with_corr_higher_than = 1

bias = 0

slope = 1

target_feat = "y5"

drop_feat = "next_next_week_yield",

num_week_shift = 0

alr_gen_feats = False

match_with_actual = False

add_mov_win_rolling_feats = False

match_with_actual_yield = False

intel_feat_moving = True

num_procs = 10