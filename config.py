#!/usr/bin/env python3

from os.path import expanduser, join
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC, KMeansSMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.combine import SMOTEENN, SMOTETomek

HOME = '/home/user'
# api network location
api_port = "4242"
base_url = "http://localhost:" + api_port

# Analysis config
apply_oversample = False

# App dirs
user_home = expanduser("~")
tfm_dir = user_home + "/src/TFM"

# Malware dirs
malware_dir = tfm_dir + "/VIRUSSHARE_LINUX/FILTERED"
malware_full_dir = malware_dir + "/1_ELF_files_uniq"
malware_arm_dir = malware_dir + "/2_ONLY_ARM"
malware_non_arm_dir = malware_dir + "/3_Non_ARM"

# Results dir
reports_dir = tfm_dir + "/Reports"
reports_success_dir = reports_dir + "/SUCCESS"
reports_failed_dir = reports_dir + "/FAILED"
reports_unknown_dir = reports_dir + "/UNKNOWN"
vt_reports_dir = join(reports_dir, 'vt_reports')
clamscan_report_filepath = HOME + "/src/TFM/Reports/Clamscan/virusshare_linux_arm_clamscan_filtered.txt"
master_db_filepath = 'master_db.pkl'

# default_exec_time = "60"
default_exec_time = "20"
dataset_size_max = '300000'

vt_waitting_time = 120
vt_max_files_per_day = 990
VT_API_KEY = '<your_api_key_here>'

post_queries = {
    "submit_full_analysis": "/api/tasks/create/file",
    # Creates full binary analysis task.
    "submit_pcap_analysis": "/api/tasks/create/pcap",
    # Creates pcap analysis task.
}

get_queries = {

    "finished": "/api/tasks/finished",
    # Lists successfully finished tasks.
    "failed": "/api/tasks/failed",
    # Lists failed tasks.
    "pending": "/api/tasks/pending",
    # Lists enqueued pending tasks.

    "task_id_view": "/api/tasks/view/<task_id>",
    # Returns tasks status.
    "task_id_report": "/api/report/<task_id>",
    # Returns analysis report.
    "task_id_pcap": "/api/pcap/<task_id>",
    # Returns pcap captured during analysis.
    "task_id_machinelog": "/api/machinelog/<task_id>",
    # Returns QEMU machinelog.
    "task_id_output": "/api/output/<task_id>",
    # Returns analyzed program's stdout output.

}

# 2: Loader config
lisa_db_filepath = "lisa_db.pkl"
vt_db_filepath = "vt_db.txt"
original_master_db_filepath = 'original_master_db.pkl'
report_keys_w_raw_data = ['type', 'static_analysis',
                          'dynamic_analysis',
                          'network_analysis',
                          'pcap',
                          'machinelog', 'output'
                          ]
essential_keys = ['file_name', 'clamscan_tag']

# Old report_keys_to_not_vectorize
keys_wo_data = [
    'type', 'vt_report', 'exec_time',
    'timestamp', 'md5', 'sha256', 'sha1',
    'analysis_start_time', 'binary_info_size',

    'binary_info_arch',
    'binary_info_os',
    'view'
]

dword_universe_filepath = 'word_universe.pkl'
dword_universe_flatten_filepath = 'word_universe_flatten.pkl'
normalized_master_db_filepath = "normalized_master_db.pkl"
master_db_minimum_filter_filepath = 'master_db_minimum_filter.pkl'
train_master_db_tfidf_filter_filepath = 'train_master_db_tfidf_filter.pkl'
dword_universe_minimum_filter_filepath = 'dword_universe_minimum_filter.pkl'
master_y_filepath = 'y.pkl'
clamav_y_filepath = 'y_clamav.pkl'
avclass_y_filepath = 'avclass_y.pkl'
avclass_train_y_filepath = 'avclass_train_y.pkl'
avclass_test_y_filepath = 'avclass_test_y.pkl'
train_master_db_filepath = 'train_master_db.pkl'
test_master_db_filepath = 'test_master_db.pkl'
dword_universe_train_X_filepath = 'dword_universe_train_X.pkl'
df_splitted_filepath = 'df_splitted.pkl'

# 4.- matrix2df
df_filepath = 'df.pkl'
target_tag = 'avclass_tag'
test_size = 0.20
min_samples_per_class = 33
word_presence_for_minimal_class_index = 0.25
maximum_amount_of_tfidf_features = 5000
min_tfidf_value = 0.02

# Minimum ammount of words appearances in the whole dataset.
# min_word_appearances = 2
min_word_appearances = min_samples_per_class * \
    word_presence_for_minimal_class_index * (1 - test_size)
verbose = True

unknown_malware_tag = 'UNKNOWN'
fields_to_ignore = ['']

index_column_name = 'file_name'
persistence_dir = "persistence"
image_dir = join(persistence_dir, "screenshots")
cm_images_dir = '../../MEMORIA/TFM_2020/graphics/TFM_2020/'
image_dir = join(cm_images_dir, "confusion_matrix")


# Step 4
forbidden_chars = ['[]{}']
# amount_of_words_to_keep = 200
amount_of_words_to_keep = 20000
train_X_filepath = 'train_X.pkl'
words_tfidf_filename = "words_tfidf.pkl"
best_words_filename = 'best_words.pkl'

# Step 5
train_X_ready_filepath = 'train_X_ready.pkl'
le_filepath = 'le.pkl'
master_y_encoded_filepath = 'master_y_encoded.pkl'

# Step 6
df_stage1_filepath = 'df_stage1.pkl'
stage2_train_X_df_filepath = 'stage2_train_X_df.pkl'
stage2_test_X_df_filepath = 'stage2_test_X_df.pkl'
num_cpus = 1
num_splits = 3
# test_size = 1/num_splits
default_num_splits = 3

default_type_of_average = 'weighted'

samplers_list = [
    ('No Sampler', None),
    # OverSample
    ('Random_Sampler', RandomOverSampler(random_state=0)),
    ('SMOTE', SMOTE()),
    ('ADASYN', ADASYN()),
    # UnderSample
    ('CondensedNearestNeighbour', CondensedNearestNeighbour()),
    # Combined - Oversampling+Undersampling
    ('SMOTENN', SMOTEENN()),
    ('SMOTETomek', SMOTETomek())
]

datamodels_list = [
    ("Naive Bayes", GaussianNB()),
    ("Nearest Neighbors",    KNeighborsClassifier(3)),
    ("Decision Tree",    DecisionTreeClassifier(max_depth=50)),
    ("Random Forest",    RandomForestClassifier(max_depth=50, n_estimators=100, max_features=10)),
    ("Neural Net", MLPClassifier(max_iter=600)),
    ("AdaBoost", AdaBoostClassifier(n_estimators=400))
]

types_of_average = ['weighted', 'micro', 'macro']

# Metrics chosen to compare and get the best algorithm
chosen_average = 'weighted'
chosen_metric = 'f1'
default_scoring = 'f1_weighted'
