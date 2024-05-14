python src/full_run.py run_name=no_features features=null use_wandb=False train_cfg.sliding_window=True test_cfg.sliding_window=True
# python src/full_run.py run_name=pixel-wise extraction_strategy=pixel-wise
# python src/full_run.py run_name=random15 add_random_neg_labels=0.15
# python src/full_run.py run_name=sliding_window_train train_cfg.sliding_window=True
# python src/full_run.py run_name=sliding_window_both train_cfg.sliding_window=True test_cfg.sliding_window=True
# python src/full_run.py run_name=sliding_window_train_random15 train_cfg.sliding_window=True add_random_neg_labels=0.15
# python src/full_run.py run_name=sliding_window_both_random15 train_cfg.sliding_window=True test_cfg.sliding_window=True add_random_neg_labels=0.15