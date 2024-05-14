# sbatch scripts/train_cluster.sh 3x3
# sbatch scripts/train_cluster.sh pixel-wise extraction_strategy=pixel-wise
# sbatch scripts/train_cluster.sh random15 add_random_neg_labels=0.15
# sbatch scripts/train_cluster.sh labels_123 labels_to_keep=[1,2,3]
# sbatch scripts/train_cluster.sh all_labels labels_to_keep=all
# sbatch scripts/train_cluster.sh sliding_window train_cfg.sliding_window=True test_cfg.sliding_window=True
# sbatch scripts/train_cluster.sh sliding_window_random05 train_cfg.sliding_window=True test_cfg.sliding_window=True add_random_neg_labels=0.05
# sbatch scripts/train_cluster.sh sliding_window_random10 train_cfg.sliding_window=True test_cfg.sliding_window=True add_random_neg_labels=0.1
# sbatch scripts/train_cluster.sh sliding_window_labels_123 train_cfg.sliding_window=True test_cfg.sliding_window=True labels_to_keep=[1,2,3]
# sbatch scripts/train_cluster.sh sliding_window_all_labels train_cfg.sliding_window=True test_cfg.sliding_window=True labels_to_keep=all

sbatch scripts/train_cluster.sh no_features features=null
sbatch scripts/train_cluster.sh no_features_sliding features=null train_cfg.sliding_window=True test_cfg.sliding_window=True