" This file contains the paths of input/output, saved in a dictionary for all datasets"

Experiments = {
  "winetype": {
    "input": "datasets/winetype.csv",
    "model_input": "models_train_test/winetype/model_winetype.pkl",
    "train_dt_input": "models_train_test/winetype/train_data_winetype.csv",
    "test_dt_input": "models_train_test/winetype/test_data_winetype.csv",
    "scores_input": "models_train_test/winetype/scores_training_winetype.csv",
    "tprfpr_input": "models_train_test/winetype/tprfpr_winetype.csv",

    "output": "results/winetype_%s.csv",
    "out_model": "models_train_test/winetype/model_winetype.pkl",
    "out_train_dt": "models_train_test/winetype/train_data_winetype.csv",
    "out_test_dt": "models_train_test/winetype/test_data_winetype.csv",
    "out_scores": "models_train_test/winetype/scores_training_winetype.csv",
    "output_tprfpr": "models_train_test/winetype/tprfpr_winetype.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "winequality": {
    "input": "datasets/winequality.csv",
    "model_input": "models_train_test/winequality/model_winequality.pkl",
    "train_dt_input": "models_train_test/winequality/train_data_winequality.csv",
    "test_dt_input": "models_train_test/winequality/test_data_winequality.csv",
    "scores_input": "models_train_test/winequality/scores_training_winequality.csv",
    "tprfpr_input": "models_train_test/winequality/tprfpr_winequality.csv",
    

    "output": "results/winequality_%s.csv",
    "out_model": "models_train_test/winequality/model_winequality.pkl",
    "out_train_dt": "models_train_test/winequality/train_data_winequality.csv",
    "out_test_dt": "models_train_test/winequality/test_data_winequality.csv",
    "out_scores": "models_train_test/winequality/scores_training_winequality.csv",
    "output_tprfpr": "models_train_test/winequality/tprfpr_winequality.csv",

    "class_feature": "class",
    "positive_label": 1,     #lower class = 1 higher class(quality)=0
    "negative_labels": None, # all other labels
    "features": None,
  },
  
  "bng": {
    "input": "datasets/BNG.csv",
    "model_input": "models_train_test/bng/model_bng.pkl",
    "train_dt_input": "models_train_test/bng/train_data_bng.csv",
    "test_dt_input": "models_train_test/bng/test_data_bng.csv",
    "scores_input": "models_train_test/bng/scores_training_bng.csv",
    "tprfpr_input": "models_train_test/bng/tprfpr_bng.csv",

    "output": "results/bng_%s.csv",
    "out_model": "models_train_test/bng/model_bng.pkl",
    "out_train_dt": "models_train_test/bng/train_data_bng.csv",
    "out_test_dt": "models_train_test/bng/test_data_bng.csv",
    "out_scores": "models_train_test/bng/scores_training_bng.csv",
    "output_tprfpr": "models_train_test/bng/tprfpr_bng.csv",

    "class_feature": "class",
    "positive_label": 2,   #we need to choose where positive class is less than total instances
    "negative_labels": None,
    "features": ["V1","V2","V3","V4","V5","V6","V7","V8","V9"],
  },

   "letter": {
    "input": "datasets/letter_recognition.csv",
    "model_input": "models_train_test/letter_recognition/model_letter_recognition.pkl",
    "train_dt_input": "models_train_test/letter_recognition/train_data_letter_recognition.csv",
    "test_dt_input": "models_train_test/letter_recognition/test_data_letter_recognition.csv",
    "scores_input": "models_train_test/letter_recognition/scores_training_letter_recognition.csv",
    "tprfpr_input": "models_train_test/letter_recognition/tprfpr_letter_recognition.csv",

    "output": "results/letter_recognition_%s.csv",
    "out_model": "models_train_test/letter_recognition/model_letter_recognition.pkl",
    "out_train_dt": "models_train_test/letter_recognition/train_data_letter_recognition.csv",
    "out_test_dt": "models_train_test/letter_recognition/test_data_letter_recognition.csv",
    "out_scores": "models_train_test/letter_recognition/scores_training_letter_recognition.csv",
    "output_tprfpr": "models_train_test/letter_recognition/tprfpr_letter_recognition.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels  
    "features": None,
  },
  "pulsar": {
    "input": "datasets/HTRU_2.csv",
    "model_input": "models_train_test/HTRU_2/model_HTRU_2.pkl",
    "train_dt_input": "models_train_test/HTRU_2/train_data_HTRU_2.csv",
    "test_dt_input": "models_train_test/HTRU_2/test_data_HTRU_2.csv",
    "scores_input": "models_train_test/HTRU_2/scores_training_HTRU_2.csv",
    "tprfpr_input": "models_train_test/HTRU_2/tprfpr_HTRU_2.csv",

    "output": "results/HTRU_2_%s.csv",
    "out_model": "models_train_test/HTRU_2/model_HTRU_2.pkl",
    "out_train_dt": "models_train_test/HTRU_2/train_data_HTRU_2.csv",
    "out_test_dt": "models_train_test/HTRU_2/test_data_HTRU_2.csv",
    "out_scores": "models_train_test/HTRU_2/scores_training_HTRU_2.csv",
    "output_tprfpr": "models_train_test/HTRU_2/tprfpr_HTRU_2.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "anurancalls": {
    "input": "datasets/anuranCalls.csv",
    "model_input": "models_train_test/anurancalls/model_anurancalls.pkl",
    "train_dt_input": "models_train_test/anurancalls/train_data_anurancalls.csv",
    "test_dt_input": "models_train_test/anurancalls/test_data_anurancalls.csv",
    "scores_input": "models_train_test/anurancalls/scores_training_anurancalls.csv",
    "tprfpr_input": "models_train_test/anurancalls/tprfpr_anurancalls.csv",

    "output": "results/anurancalls_%s.csv",
    "out_model": "models_train_test/anurancalls/model_anurancalls.pkl",
    "out_train_dt": "models_train_test/anurancalls/train_data_anurancalls.csv",
    "out_test_dt": "models_train_test/anurancalls/test_data_anurancalls.csv",
    "out_scores": "models_train_test/anurancalls/scores_training_anurancalls.csv",
    "output_tprfpr": "models_train_test/anurancalls/tprfpr_anurancalls.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None,
    "features": ['MFCCs_1','MFCCs_2','MFCCs_3','MFCCs_4','MFCCs_5','MFCCs_6','MFCCs_7','MFCCs_8','MFCCs_9','MFCCs_10','MFCCs_11','MFCCs_12','MFCCs_13','MFCCs_14','MFCCs_15','MFCCs_16','MFCCs_17','MFCCs_18','MFCCs_19','MFCCs_20','MFCCs_21','MFCCs_22'],
  },
  "handwritten": {
    "input": "datasets/Handwritten.csv",
    "model_input": "models_train_test/handwritten/model_handwritten.pkl",
    "train_dt_input": "models_train_test/handwritten/train_data_handwritten.csv",
    "test_dt_input": "models_train_test/handwritten/test_data_handwritten.csv",
    "scores_input": "models_train_test/handwritten/scores_training_handwritten.csv",
    "tprfpr_input": "models_train_test/handwritten/tprfpr_handwritten.csv",

    "output": "results/handwritten_%s.csv",
    "out_model": "models_train_test/handwritten/model_handwritten.pkl",
    "out_train_dt": "models_train_test/handwritten/train_data_handwritten.csv",
    "out_test_dt": "models_train_test/handwritten/test_data_handwritten.csv",
    "out_scores": "models_train_test/handwritten/scores_training_handwritten.csv",
    "output_tprfpr": "models_train_test/handwritten/tprfpr_handwritten.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": lambda x: x != 'letter' and x != 'author',
  },
  "arabic": {
    "input": "datasets/ArabicDigit.csv",
    "model_input": "models_train_test/arabicdigit/model_arabicdigit.pkl",
    "train_dt_input": "models_train_test/arabicdigit/train_data_arabicdigit.csv",
    "test_dt_input": "models_train_test/arabicdigit/test_data_arabicdigit.csv",
    "scores_input": "models_train_test/arabicdigit/scores_training_arabicdigit.csv",
    "tprfpr_input": "models_train_test/arabicdigit/tprfpr_arabicdigit.csv",

    "output": "results/arabicdigit_%s.csv",
    "out_model": "models_train_test/arabicdigit/model_arabicdigit.pkl",
    "out_train_dt": "models_train_test/arabicdigit/train_data_arabicdigit.csv",
    "out_test_dt": "models_train_test/arabicdigit/test_data_arabicdigit.csv",
    "out_scores": "models_train_test/arabicdigit/scores_training_arabicdigit.csv",
    "output_tprfpr": "models_train_test/arabicdigit/tprfpr_arabicdigit.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": lambda x: x != 'sex' and x != 'digit',
  },
  "insects": {
    "input": "datasets/AllSpecies.csv",
    "output": "models_train_test/AllSpecies_%s.csv",
    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": [1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 22, 24],
    "features": ["temperature", "wbf","eh_1","eh_2","eh_3","eh_4","eh_5","eh_6","eh_7","eh_8","eh_9","eh_10","eh_11","eh_12","eh_13","eh_14","eh_15","eh_16","eh_17","eh_18","eh_19","eh_20","eh_21","eh_22","eh_23","eh_24","eh_25"],
  },
  "aedessex": {
    "input": "datasets/aedessex.csv",
    "model_input": "models_train_test/aedessex/model_aedessex.pkl",
    "train_dt_input": "models_train_test/aedessex/train_data_aedessex.csv",
    "test_dt_input": "models_train_test/aedessex/test_data_aedessex.csv",
    "scores_input": "models_train_test/aedessex/scores_training_aedessex.csv",
    "tprfpr_input": "models_train_test/aedessex/tprfpr_aedessex.csv",

    "output": "results/aedessex_%s.csv",
    "out_model": "models_train_test/aedessex/model_aedessex.pkl",
    "out_train_dt": "models_train_test/aedessex/train_data_aedessex.csv",
    "out_test_dt": "models_train_test/aedessex/test_data_aedessex.csv",
    "out_scores": "models_train_test/aedessex/scores_training_aedessex.csv",
    "output_tprfpr": "models_train_test/aedessex/tprfpr_aedessex.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 22, 24],
    "features": ["temperature", "wbf","eh_1","eh_2","eh_3","eh_4","eh_5","eh_6","eh_7","eh_8","eh_9","eh_10","eh_11","eh_12","eh_13","eh_14","eh_15","eh_16","eh_17","eh_18","eh_19","eh_20","eh_21","eh_22","eh_23","eh_24","eh_25"],
  },

  "aedesquinx": {
    "input": "datasets/aedesQuinx.csv",
    "model_input": "models_train_test/aedesquinx/model_aedesquinx.pkl",
    "train_dt_input": "models_train_test/aedesquinx/train_data_aedesquinx.csv",
    "test_dt_input": "models_train_test/aedesquinx/test_data_aedesquinx.csv",
    "scores_input": "models_train_test/aedesquinx/scores_training_aedesquinx.csv",
    "tprfpr_input": "models_train_test/aedesquinx/tprfpr_aedesquinx.csv",

    "output": "results/aedesquinx_%s.csv",
    "out_model": "models_train_test/aedesquinx/model_aedesquinx.pkl",
    "out_train_dt": "models_train_test/aedesquinx/train_data_aedesquinx.csv",
    "out_test_dt": "models_train_test/aedesquinx/test_data_aedesquinx.csv",
    "out_scores": "models_train_test/aedesquinx/scores_training_aedesquinx.csv",
    "output_tprfpr": "models_train_test/aedesquinx/tprfpr_aedesquinx.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 22, 24],
    "features": ["temperature", "wbf","eh_1","eh_2","eh_3","eh_4","eh_5","eh_6","eh_7","eh_8","eh_9","eh_10","eh_11","eh_12","eh_13","eh_14","eh_15","eh_16","eh_17","eh_18","eh_19","eh_20","eh_21","eh_22","eh_23","eh_24","eh_25"],
  },
  "mushroom": {
    "input": "datasets/mushroom.csv",
    "model_input": "models_train_test/mushroom/model_mushroom.pkl",
    "train_dt_input": "models_train_test/mushroom/train_data_mushroom.csv",
    "test_dt_input": "models_train_test/mushroom/test_data_mushroom.csv",
    "scores_input": "models_train_test/mushroom/scores_training_mushroom.csv",
    "tprfpr_input": "models_train_test/mushroom/tprfpr_mushroom.csv",

    "output": "results/mushroom_%s.csv",
    "out_model": "models_train_test/mushroom/model_mushroom.pkl",
    "out_train_dt": "models_train_test/mushroom/train_data_mushroom.csv",
    "out_test_dt": "models_train_test/mushroom/test_data_mushroom.csv",
    "out_scores": "models_train_test/mushroom/scores_training_mushroom.csv",
    "output_tprfpr": "models_train_test/mushroom/tprfpr_mushroom.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },

  "spambase": {
    "input": "datasets/spambase.csv",
    "model_input": "models_train_test/spambase/model_spambase.pkl",
    "train_dt_input": "models_train_test/spambase/train_data_spambase.csv",
    "test_dt_input": "models_train_test/spambase/test_data_spambase.csv",
    "scores_input": "models_train_test/spambase/scores_training_spambase.csv",
    "tprfpr_input": "models_train_test/spambase/tprfpr_spambase.csv",
    
    "output": "results/spambase_%s.csv",
    "out_model": "models_train_test/spambase/model_spambase.pkl",
    "out_train_dt": "models_train_test/spambase/train_data_spambase.csv",
    "out_test_dt": "models_train_test/spambase/test_data_spambase.csv",
    "out_scores": "models_train_test/spambase/scores_training_spambase.csv",
    "output_tprfpr": "models_train_test/spambase/tprfpr_spambase.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "mozilla": {
    "input": "datasets/mozilla.csv",
    "model_input": "models_train_test/mozilla/model_mozilla.pkl",
    "train_dt_input": "models_train_test/mozilla/train_data_mozilla.csv",
    "test_dt_input": "models_train_test/mozilla/test_data_mozilla.csv",
    "scores_input": "models_train_test/mozilla/scores_training_mozilla.csv",
    "tprfpr_input": "models_train_test/mozilla/tprfpr_mozilla.csv",

    "output": "results/mozilla_%s.csv",
    "out_model": "models_train_test/mozilla/model_mozilla.pkl",
    "out_train_dt": "models_train_test/mozilla/train_data_mozilla.csv",
    "out_test_dt": "models_train_test/mozilla/test_data_mozilla.csv",
    "out_scores": "models_train_test/mozilla/scores_training_mozilla.csv",
    "output_tprfpr": "models_train_test/mozilla/tprfpr_mozilla.csv",
    
    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "namao": {
    "input": "datasets/namao.csv",
    "model_input": "models_train_test/namao/model_namao.pkl",
    "train_dt_input": "models_train_test/namao/train_data_namao.csv",
    "test_dt_input": "models_train_test/namao/test_data_namao.csv",
    "scores_input": "models_train_test/namao/scores_training_namao.csv",
    "tprfpr_input": "models_train_test/namao/tprfpr_namao.csv",
    
    "output": "results/namao_%s.csv",
    "out_model": "models_train_test/namao/model_namao.pkl",
    "out_train_dt": "models_train_test/namao/train_data_namao.csv",
    "out_test_dt": "models_train_test/namao/test_data_namao.csv",
    "out_scores": "models_train_test/namao/scores_training_namao.csv",
    "output_tprfpr": "models_train_test/namao/tprfpr_namao.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "occupancy": {
    "input": "datasets/occupancy.csv",
    "model_input": "models_train_test/occupancy/model_occupancy.pkl",
    "train_dt_input": "models_train_test/occupancy/train_data_occupancy.csv",
    "test_dt_input": "models_train_test/occupancy/test_data_occupancy.csv",
    "scores_input": "models_train_test/occupancy/scores_training_occupancy.csv",
    "tprfpr_input": "models_train_test/occupancy/tprfpr_occupancy.csv",

    "output": "results/occupancy_%s.csv",
    "out_model": "models_train_test/occupancy/model_occupancy.pkl",
    "out_train_dt": "models_train_test/occupancy/train_data_occupancy.csv",
    "out_test_dt": "models_train_test/occupancy/test_data_occupancy.csv",
    "out_scores": "models_train_test/occupancy/scores_training_occupancy.csv",
    "output_tprfpr": "models_train_test/occupancy/tprfpr_occupancy.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "eeg": {
    "input": "datasets/eeg.csv",
    "model_input": "models_train_test/eeg/model_eeg.pkl",
    "train_dt_input": "models_train_test/eeg/train_data_eeg.csv",
    "test_dt_input": "models_train_test/eeg/test_data_eeg.csv",
    "scores_input": "models_train_test/eeg/scores_training_eeg.csv",
    "tprfpr_input": "models_train_test/eeg/tprfpr_eeg.csv",

    "output": "results/eeg_%s.csv",
    "out_model": "models_train_test/eeg/model_eeg.pkl",
    "out_train_dt": "models_train_test/eeg/train_data_eeg.csv",
    "out_test_dt": "models_train_test/eeg/test_data_eeg.csv",
    "out_scores": "models_train_test/eeg/scores_training_eeg.csv",
    "output_tprfpr": "models_train_test/eeg/tprfpr_eeg.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "click": {
    "input": "datasets/click_prediction.csv",
    "model_input": "models_train_test/click_prediction/model_click_prediction.pkl",
    "train_dt_input": "models_train_test/click_prediction/train_data_click_prediction.csv",
    "test_dt_input": "models_train_test/click_prediction/test_data_click_prediction.csv",
    "scores_input": "models_train_test/click_prediction/scores_training_click_prediction.csv",
    "tprfpr_input": "models_train_test/click_prediction/tprfpr_click_prediction.csv",

    "output": "results/click_prediction_%s.csv",
    "out_model": "models_train_test/click_prediction/model_click_prediction.pkl",
    "out_train_dt": "models_train_test/click_prediction/train_data_click_prediction.csv",
    "out_test_dt": "models_train_test/click_prediction/test_data_click_prediction.csv",
    "out_scores": "models_train_test/click_prediction/scores_training_click_prediction.csv",
    "output_tprfpr": "models_train_test/click_prediction/tprfpr_click_prediction.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "pollen": {
    "input": "datasets/pollen.csv",
    "model_input": "models_train_test/pollen/model_pollen.pkl",
    "train_dt_input": "models_train_test/pollen/train_data_pollen.csv",
    "test_dt_input": "models_train_test/pollen/test_data_pollen.csv",
    "scores_input": "models_train_test/pollen/scores_training_pollen.csv",
    "tprfpr_input": "models_train_test/pollen/tprfpr_pollen.csv",

    "output": "results/pollen_%s.csv",
    "out_model": "models_train_test/pollen/model_pollen.pkl",
    "out_train_dt": "models_train_test/pollen/train_data_pollen.csv",
    "out_test_dt": "models_train_test/pollen/test_data_pollen.csv",
    "out_scores": "models_train_test/pollen/scores_training_pollen.csv",
    "output_tprfpr": "models_train_test/pollen/tprfpr_pollen.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "bank": {
    "input": "datasets/bank.csv",
    "model_input": "models_train_test/bank/model_bank.pkl",
    "train_dt_input": "models_train_test/bank/train_data_bank.csv",
    "test_dt_input": "models_train_test/bank/test_data_bank.csv",
    "scores_input": "models_train_test/bank/scores_training_bank.csv",
    "tprfpr_input": "models_train_test/bank/tprfpr_bank.csv",

    "output": "results/bank_%s.csv",
    "out_model": "models_train_test/bank/model_bank.pkl",
    "out_train_dt": "models_train_test/bank/train_data_bank.csv",
    "out_test_dt": "models_train_test/bank/test_data_bank.csv",
    "out_scores": "models_train_test/bank/scores_training_bank.csv",
    "output_tprfpr": "models_train_test/bank/tprfpr_bank.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "cmc": {
    "input": "datasets/cmc.csv",
    "model_input": "models_train_test/cmc/model_cmc.pkl",
    "train_dt_input": "models_train_test/cmc/train_data_cmc.csv",
    "test_dt_input": "models_train_test/cmc/test_data_cmc.csv",
    "scores_input": "models_train_test/cmc/scores_training_cmc.csv",
    "tprfpr_input": "models_train_test/cmc/tprfpr_cmc.csv",

    "output": "results/cmc_%s.csv",
    "out_model": "models_train_test/cmc/model_cmc.pkl",
    "out_train_dt": "models_train_test/cmc/train_data_cmc.csv",
    "out_test_dt": "models_train_test/cmc/test_data_cmc.csv",
    "out_scores": "models_train_test/cmc/scores_training_cmc.csv",
    "output_tprfpr": "models_train_test/cmc/tprfpr_cmc.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "covariance": {
    "input": "datasets/covtype_reduced.csv",
    "model_input": "models_train_test/covtype_reduced/model_covtype_reduced.pkl",
    "train_dt_input": "models_train_test/covtype_reduced/train_data_covtype_reduced.csv",
    "test_dt_input": "models_train_test/covtype_reduced/test_data_covtype_reduced.csv",
    "scores_input": "models_train_test/covtype_reduced/scores_training_covtype_reduced.csv",
    "tprfpr_input": "models_train_test/covtype_reduced/tprfpr_covtype_reduced.csv",

    "output": "results/covtype_reduced_%s.csv",
    "out_model": "models_train_test/covtype_reduced/model_covtype_reduced.pkl",
    "out_train_dt": "models_train_test/covtype_reduced/train_data_covtype_reduced.csv",
    "out_test_dt": "models_train_test/covtype_reduced/test_data_covtype_reduced.csv",
    "out_scores": "models_train_test/covtype_reduced/scores_training_covtype_reduced.csv",
    "output_tprfpr": "models_train_test/covtype_reduced/tprfpr_covtype_reduced.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "credit": {
    "input": "datasets/default_credit_card_clients.csv",
    "model_input": "models_train_test/default_credit_card_clients/model_default_credit_card_clients.pkl",
    "train_dt_input": "models_train_test/default_credit_card_clients/train_data_default_credit_card_clients.csv",
    "test_dt_input": "models_train_test/default_credit_card_clients/test_data_default_credit_card_clients.csv",
    "scores_input": "models_train_test/default_credit_card_clients/scores_training_default_credit_card_clients.csv",
    "tprfpr_input": "models_train_test/default_credit_card_clients/tprfpr_default_credit_card_clients.csv",

    "output": "results/default_credit_card_clients_%s.csv",
    "out_model": "models_train_test/default_credit_card_clients/model_default_credit_card_clients.pkl",
    "out_train_dt": "models_train_test/default_credit_card_clients/train_data_default_credit_card_clients.csv",
    "out_test_dt": "models_train_test/default_credit_card_clients/test_data_default_credit_card_clients.csv",
    "out_scores": "models_train_test/default_credit_card_clients/scores_training_default_credit_card_clients.csv",
    "output_tprfpr": "models_train_test/default_credit_card_clients/tprfpr_default_credit_card_clients.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "magic": {
    "input": "datasets/magic.csv",
    "model_input": "models_train_test/magic/model_magic.pkl",
    "train_dt_input": "models_train_test/magic/train_data_magic.csv",
    "test_dt_input": "models_train_test/magic/test_data_magic.csv",
    "scores_input": "models_train_test/magic/scores_training_magic.csv",
    "tprfpr_input": "models_train_test/magic/tprfpr_magic.csv",

    "output": "results/magic_%s.csv",
    "out_model": "models_train_test/magic/model_magic.pkl",
    "out_train_dt": "models_train_test/magic/train_data_magic.csv",
    "out_test_dt": "models_train_test/magic/test_data_magic.csv",
    "out_scores": "models_train_test/magic/scores_training_magic.csv",
    "output_tprfpr": "models_train_test/magic/tprfpr_magic.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "jm1": {
    "input": "datasets/jm1_processed.csv",
    "model_input": "models_train_test/jm1_processed/model_jm1_processed.pkl",
    "train_dt_input": "models_train_test/jm1_processed/train_data_jm1_processed.csv",
    "test_dt_input": "models_train_test/jm1_processed/test_data_jm1_processed.csv",
    "scores_input": "models_train_test/jm1_processed/scores_training_jm1_processed.csv",
    "tprfpr_input": "models_train_test/jm1_processed/tprfpr_jm1_processed.csv",

    "output": "results/jm1_processed_%s.csv",
    "out_model": "models_train_test/jm1_processed/model_jm1_processed.pkl",
    "out_train_dt": "models_train_test/jm1_processed/train_data_jm1_processed.csv",
    "out_test_dt": "models_train_test/jm1_processed/test_data_jm1_processed.csv",
    "out_scores": "models_train_test/jm1_processed/scores_training_jm1_processed.csv",
    "output_tprfpr": "models_train_test/jm1_processed/tprfpr_jm1_processed.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "phoneme": {
    "input": "datasets/phoneme.csv",
    "model_input": "models_train_test/phoneme/model_phoneme.pkl",
    "train_dt_input": "models_train_test/phoneme/train_data_phoneme.csv",
    "test_dt_input": "models_train_test/phoneme/test_data_phoneme.csv",
    "scores_input": "models_train_test/phoneme/scores_training_phoneme.csv",
    "tprfpr_input": "models_train_test/phoneme/tprfpr_phoneme.csv",

    "output": "results/phoneme_%s.csv",
    "out_model": "models_train_test/phoneme/model_phoneme.pkl",
    "out_train_dt": "models_train_test/phoneme/train_data_phoneme.csv",
    "out_test_dt": "models_train_test/phoneme/test_data_phoneme.csv",
    "out_scores": "models_train_test/phoneme/scores_training_phoneme.csv",
    "output_tprfpr": "models_train_test/phoneme/tprfpr_phoneme.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "stock": {
    "input": "datasets/stock_market.csv",
    "model_input": "models_train_test/stock/model_stock.pkl",
    "train_dt_input": "models_train_test/stock/train_data_stock.csv",
    "test_dt_input": "models_train_test/stock/test_data_stock.csv",
    "scores_input": "models_train_test/stock/scores_training_stock.csv",
    "tprfpr_input": "models_train_test/stock/tprfpr_stock.csv",

    "output": "results/stock_%s.csv",
    "out_model": "models_train_test/stock/model_stock.pkl",
    "out_train_dt": "models_train_test/stock/train_data_stock.csv",
    "out_test_dt": "models_train_test/stock/test_data_stock.csv",
    "out_scores": "models_train_test/stock/scores_training_stock.csv",
    "output_tprfpr": "models_train_test/stock/tprfpr_stock.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "wind": {
    "input": "datasets/wind.csv",
    "model_input": "models_train_test/wind/model_wind.pkl",
    "train_dt_input": "models_train_test/wind/train_data_wind.csv",
    "test_dt_input": "models_train_test/wind/test_data_wind.csv",
    "scores_input": "models_train_test/wind/scores_training_wind.csv",
    "tprfpr_input": "models_train_test/wind/tprfpr_wind.csv",

    "output": "results/wind_%s.csv",
    "out_model": "models_train_test/wind/model_wind.pkl",
    "out_train_dt": "models_train_test/wind/train_data_wind.csv",
    "out_test_dt": "models_train_test/wind/test_data_wind.csv",
    "out_scores": "models_train_test/wind/scores_training_wind.csv",
    "output_tprfpr": "models_train_test/wind/tprfpr_wind.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "waveform": {
    "input": "datasets/waveform.csv",
    "model_input": "models_train_test/waveform/model_waveform.pkl",
    "train_dt_input": "models_train_test/waveform/train_data_waveform.csv",
    "test_dt_input": "models_train_test/waveform/test_data_waveform.csv",
    "scores_input": "models_train_test/waveform/scores_training_waveform.csv",
    "tprfpr_input": "models_train_test/waveform/tprfpr_waveform.csv",

    "output": "results/waveform_%s.csv",
    "out_model": "models_train_test/waveform/model_waveform.pkl",
    "out_train_dt": "models_train_test/waveform/train_data_waveform.csv",
    "out_test_dt": "models_train_test/waveform/test_data_waveform.csv",
    "out_scores": "models_train_test/waveform/scores_training_waveform.csv",
    "output_tprfpr": "models_train_test/waveform/tprfpr_waveform.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "2dplanes": {
    "input": "datasets/2dplanes.csv",
    "model_input": "models_train_test/2dplanes/model_2dplanes.pkl",
    "train_dt_input": "models_train_test/2dplanes/train_data_2dplanes.csv",
    "test_dt_input": "models_train_test/2dplanes/test_data_2dplanes.csv",
    "scores_input": "models_train_test/2dplanes/scores_training_2dplanes.csv",
    "tprfpr_input": "models_train_test/2dplanes/tprfpr_2dplanes.csv",

    "output": "results/2dplanes_%s.csv",
    "out_model": "models_train_test/2dplanes/model_2dplanes.pkl",
    "out_train_dt": "models_train_test/2dplanes/train_data_2dplanes.csv",
    "out_test_dt": "models_train_test/2dplanes/test_data_2dplanes.csv",
    "out_scores": "models_train_test/2dplanes/scores_training_2dplanes.csv",
    "output_tprfpr": "models_train_test/2dplanes/tprfpr_2dplanes.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "abalone": {
    "input": "datasets/abalone.csv",
    "model_input": "models_train_test/abalone/model_abalone.pkl",
    "train_dt_input": "models_train_test/abalone/train_data_abalone.csv",
    "test_dt_input": "models_train_test/abalone/test_data_abalone.csv",
    "scores_input": "models_train_test/abalone/scores_training_abalone.csv",
    "tprfpr_input": "models_train_test/abalone/tprfpr_abalone.csv",

    "output": "results/abalone_%s.csv",
    "out_model": "models_train_test/abalone/model_abalone.pkl",
    "out_train_dt": "models_train_test/abalone/train_data_abalone.csv",
    "out_test_dt": "models_train_test/abalone/test_data_abalone.csv",
    "out_scores": "models_train_test/abalone/scores_training_abalone.csv",
    "output_tprfpr": "models_train_test/abalone/tprfpr_abalone.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "ada": {
    "input": "datasets/ada.csv",
    "model_input": "models_train_test/ada/model_ada.pkl",
    "train_dt_input": "models_train_test/ada/train_data_ada.csv",
    "test_dt_input": "models_train_test/ada/test_data_ada.csv",
    "scores_input": "models_train_test/ada/scores_training_ada.csv",
    "tprfpr_input": "models_train_test/ada/tprfpr_ada.csv",

    "output": "results/ada_%s.csv",
    "out_model": "models_train_test/ada/model_ada.pkl",
    "out_train_dt": "models_train_test/ada/train_data_ada.csv",
    "out_test_dt": "models_train_test/ada/test_data_ada.csv",
    "out_scores": "models_train_test/ada/scores_training_ada.csv",
    "output_tprfpr": "models_train_test/ada/tprfpr_ada.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "ada_prior": {
    "input": "datasets/ada_prior.csv",
    "model_input": "models_train_test/ada_prior/model_ada_prior.pkl",
    "train_dt_input": "models_train_test/ada_prior/train_data_ada_prior.csv",
    "test_dt_input": "models_train_test/ada_prior/test_data_ada_prior.csv",
    "scores_input": "models_train_test/ada_prior/scores_training_ada_prior.csv",
    "tprfpr_input": "models_train_test/ada_prior/tprfpr_ada_prior.csv",

    "output": "results/ada_prior_%s.csv",
    "out_model": "models_train_test/ada_prior/model_ada_prior.pkl",
    "out_train_dt": "models_train_test/ada_prior/train_data_ada_prior.csv",
    "out_test_dt": "models_train_test/ada_prior/test_data_ada_prior.csv",
    "out_scores": "models_train_test/ada_prior/scores_training_ada_prior.csv",
    "output_tprfpr": "models_train_test/ada_prior/tprfpr_ada_prior.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "adult": {
    "input": "datasets/adult.csv",
    "model_input": "models_train_test/adult/model_adult.pkl",
    "train_dt_input": "models_train_test/adult/train_data_adult.csv",
    "test_dt_input": "models_train_test/adult/test_data_adult.csv",
    "scores_input": "models_train_test/adult/scores_training_adult.csv",
    "tprfpr_input": "models_train_test/adult/tprfpr_adult.csv",

    "output": "results/adult_%s.csv",
    "out_model": "models_train_test/adult/model_adult.pkl",
    "out_train_dt": "models_train_test/adult/train_data_adult.csv",
    "out_test_dt": "models_train_test/adult/test_data_adult.csv",
    "out_scores": "models_train_test/adult/scores_training_adult.csv",
    "output_tprfpr": "models_train_test/adult/tprfpr_adult.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "adult_1590": {
    "input": "datasets/adult_1590.csv",
    "model_input": "models_train_test/adult_1590/model_adult_1590.pkl",
    "train_dt_input": "models_train_test/adult_1590/train_data_adult_1590.csv",
    "test_dt_input": "models_train_test/adult_1590/test_data_adult_1590.csv",
    "scores_input": "models_train_test/adult_1590/scores_training_adult_1590.csv",
    "tprfpr_input": "models_train_test/adult_1590/tprfpr_adult_1590.csv",

    "output": "results/adult_1590_%s.csv",
    "out_model": "models_train_test/adult_1590/model_adult_1590.pkl",
    "out_train_dt": "models_train_test/adult_1590/train_data_adult_1590.csv",
    "out_test_dt": "models_train_test/adult_1590/test_data_adult_1590.csv",
    "out_scores": "models_train_test/adult_1590/scores_training_adult_1590.csv",
    "output_tprfpr": "models_train_test/adult_1590/tprfpr_adult_1590.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "adult_census": {
    "input": "datasets/adult_census.csv",
    "model_input": "models_train_test/adult_census/model_adult_census.pkl",
    "train_dt_input": "models_train_test/adult_census/train_data_adult_census.csv",
    "test_dt_input": "models_train_test/adult_census/test_data_adult_census.csv",
    "scores_input": "models_train_test/adult_census/scores_training_adult_census.csv",
    "tprfpr_input": "models_train_test/adult_census/tprfpr_adult_census.csv",

    "output": "results/adult_census_%s.csv",
    "out_model": "models_train_test/adult_census/model_adult_census.pkl",
    "out_train_dt": "models_train_test/adult_census/train_data_adult_census.csv",
    "out_test_dt": "models_train_test/adult_census/test_data_adult_census.csv",
    "out_scores": "models_train_test/adult_census/scores_training_adult_census.csv",
    "output_tprfpr": "models_train_test/adult_census/tprfpr_adult_census.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "ailerons": {
    "input": "datasets/ailerons.csv",
    "model_input": "models_train_test/ailerons/model_ailerons.pkl",
    "train_dt_input": "models_train_test/ailerons/train_data_ailerons.csv",
    "test_dt_input": "models_train_test/ailerons/test_data_ailerons.csv",
    "scores_input": "models_train_test/ailerons/scores_training_ailerons.csv",
    "tprfpr_input": "models_train_test/ailerons/tprfpr_ailerons.csv",

    "output": "results/ailerons_%s.csv",
    "out_model": "models_train_test/ailerons/model_ailerons.pkl",
    "out_train_dt": "models_train_test/ailerons/train_data_ailerons.csv",
    "out_test_dt": "models_train_test/ailerons/test_data_ailerons.csv",
    "out_scores": "models_train_test/ailerons/scores_training_ailerons.csv",
    "output_tprfpr": "models_train_test/ailerons/tprfpr_ailerons.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "analcatdata_supreme": {
    "input": "datasets/analcatdata_supreme.csv",
    "model_input": "models_train_test/analcatdata_supreme/model_analcatdata_supreme.pkl",
    "train_dt_input": "models_train_test/analcatdata_supreme/train_data_analcatdata_supreme.csv",
    "test_dt_input": "models_train_test/analcatdata_supreme/test_data_analcatdata_supreme.csv",
    "scores_input": "models_train_test/analcatdata_supreme/scores_training_analcatdata_supreme.csv",
    "tprfpr_input": "models_train_test/analcatdata_supreme/tprfpr_analcatdata_supreme.csv",

    "output": "results/analcatdata_supreme_%s.csv",
    "out_model": "models_train_test/analcatdata_supreme/model_analcatdata_supreme.pkl",
    "out_train_dt": "models_train_test/analcatdata_supreme/train_data_analcatdata_supreme.csv",
    "out_test_dt": "models_train_test/analcatdata_supreme/test_data_analcatdata_supreme.csv",
    "out_scores": "models_train_test/analcatdata_supreme/scores_training_analcatdata_supreme.csv",
    "output_tprfpr": "models_train_test/analcatdata_supreme/tprfpr_analcatdata_supreme.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "APSFailure": {
    "input": "datasets/APSFailure.csv",
    "model_input": "models_train_test/APSFailure/model_APSFailure.pkl",
    "train_dt_input": "models_train_test/APSFailure/train_data_APSFailure.csv",
    "test_dt_input": "models_train_test/APSFailure/test_data_APSFailure.csv",
    "scores_input": "models_train_test/APSFailure/scores_training_APSFailure.csv",
    "tprfpr_input": "models_train_test/APSFailure/tprfpr_APSFailure.csv",

    "output": "results/APSFailure_%s.csv",
    "out_model": "models_train_test/APSFailure/model_APSFailure.pkl",
    "out_train_dt": "models_train_test/APSFailure/train_data_APSFailure.csv",
    "out_test_dt": "models_train_test/APSFailure/test_data_APSFailure.csv",
    "out_scores": "models_train_test/APSFailure/scores_training_APSFailure.csv",
    "output_tprfpr": "models_train_test/APSFailure/tprfpr_APSFailure.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "balloon": {
    "input": "datasets/balloon.csv",
    "model_input": "models_train_test/balloon/model_balloon.pkl",
    "train_dt_input": "models_train_test/balloon/train_data_balloon.csv",
    "test_dt_input": "models_train_test/balloon/test_data_balloon.csv",
    "scores_input": "models_train_test/balloon/scores_training_balloon.csv",
    "tprfpr_input": "models_train_test/balloon/tprfpr_balloon.csv",

    "output": "results/balloon_%s.csv",
    "out_model": "models_train_test/balloon/model_balloon.pkl",
    "out_train_dt": "models_train_test/balloon/train_data_balloon.csv",
    "out_test_dt": "models_train_test/balloon/test_data_balloon.csv",
    "out_scores": "models_train_test/balloon/scores_training_balloon.csv",
    "output_tprfpr": "models_train_test/balloon/tprfpr_balloon.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "banana": {
    "input": "datasets/banana.csv",
    "model_input": "models_train_test/banana/model_banana.pkl",
    "train_dt_input": "models_train_test/banana/train_data_banana.csv",
    "test_dt_input": "models_train_test/banana/test_data_banana.csv",
    "scores_input": "models_train_test/banana/scores_training_banana.csv",
    "tprfpr_input": "models_train_test/banana/tprfpr_banana.csv",

    "output": "results/banana_%s.csv",
    "out_model": "models_train_test/banana/model_banana.pkl",
    "out_train_dt": "models_train_test/banana/train_data_banana.csv",
    "out_test_dt": "models_train_test/banana/test_data_banana.csv",
    "out_scores": "models_train_test/banana/scores_training_banana.csv",
    "output_tprfpr": "models_train_test/banana/tprfpr_banana.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "bank_marketing": {
    "input": "datasets/bank_marketing.csv",
    "model_input": "models_train_test/bank_marketing/model_bank_marketing.pkl",
    "train_dt_input": "models_train_test/bank_marketing/train_data_bank_marketing.csv",
    "test_dt_input": "models_train_test/bank_marketing/test_data_bank_marketing.csv",
    "scores_input": "models_train_test/bank_marketing/scores_training_bank_marketing.csv",
    "tprfpr_input": "models_train_test/bank_marketing/tprfpr_bank_marketing.csv",

    "output": "results/bank_marketing_%s.csv",
    "out_model": "models_train_test/bank_marketing/model_bank_marketing.pkl",
    "out_train_dt": "models_train_test/bank_marketing/train_data_bank_marketing.csv",
    "out_test_dt": "models_train_test/bank_marketing/test_data_bank_marketing.csv",
    "out_scores": "models_train_test/bank_marketing/scores_training_bank_marketing.csv",
    "output_tprfpr": "models_train_test/bank_marketing/tprfpr_bank_marketing.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "bank8FM": {
    "input": "datasets/bank8FM.csv",
    "model_input": "models_train_test/bank8FM/model_bank8FM.pkl",
    "train_dt_input": "models_train_test/bank8FM/train_data_bank8FM.csv",
    "test_dt_input": "models_train_test/bank8FM/test_data_bank8FM.csv",
    "scores_input": "models_train_test/bank8FM/scores_training_bank8FM.csv",
    "tprfpr_input": "models_train_test/bank8FM/tprfpr_bank8FM.csv",

    "output": "results/bank8FM_%s.csv",
    "out_model": "models_train_test/bank8FM/model_bank8FM.pkl",
    "out_train_dt": "models_train_test/bank8FM/train_data_bank8FM.csv",
    "out_test_dt": "models_train_test/bank8FM/test_data_bank8FM.csv",
    "out_scores": "models_train_test/bank8FM/scores_training_bank8FM.csv",
    "output_tprfpr": "models_train_test/bank8FM/tprfpr_bank8FM.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "bank32nh": {
    "input": "datasets/bank32nh.csv",
    "model_input": "models_train_test/bank32nh/model_bank32nh.pkl",
    "train_dt_input": "models_train_test/bank32nh/train_data_bank32nh.csv",
    "test_dt_input": "models_train_test/bank32nh/test_data_bank32nh.csv",
    "scores_input": "models_train_test/bank32nh/scores_training_bank32nh.csv",
    "tprfpr_input": "models_train_test/bank32nh/tprfpr_bank32nh.csv",

    "output": "results/bank32nh_%s.csv",
    "out_model": "models_train_test/bank32nh/model_bank32nh.pkl",
    "out_train_dt": "models_train_test/bank32nh/train_data_bank32nh.csv",
    "out_test_dt": "models_train_test/bank32nh/test_data_bank32nh.csv",
    "out_scores": "models_train_test/bank32nh/scores_training_bank32nh.csv",
    "output_tprfpr": "models_train_test/bank32nh/tprfpr_bank32nh.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "banknote_authentication": {
    "input": "datasets/banknote_authentication.csv",
    "model_input": "models_train_test/banknote_authentication/model_banknote_authentication.pkl",
    "train_dt_input": "models_train_test/banknote_authentication/train_data_banknote_authentication.csv",
    "test_dt_input": "models_train_test/banknote_authentication/test_data_banknote_authentication.csv",
    "scores_input": "models_train_test/banknote_authentication/scores_training_banknote_authentication.csv",
    "tprfpr_input": "models_train_test/banknote_authentication/tprfpr_banknote_authentication.csv",

    "output": "results/banknote_authentication_%s.csv",
    "out_model": "models_train_test/banknote_authentication/model_banknote_authentication.pkl",
    "out_train_dt": "models_train_test/banknote_authentication/train_data_banknote_authentication.csv",
    "out_test_dt": "models_train_test/banknote_authentication/test_data_banknote_authentication.csv",
    "out_scores": "models_train_test/banknote_authentication/scores_training_banknote_authentication.csv",
    "output_tprfpr": "models_train_test/banknote_authentication/tprfpr_banknote_authentication.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "car": {
    "input": "datasets/car.csv",
    "model_input": "models_train_test/car/model_car.pkl",
    "train_dt_input": "models_train_test/car/train_data_car.csv",
    "test_dt_input": "models_train_test/car/test_data_car.csv",
    "scores_input": "models_train_test/car/scores_training_car.csv",
    "tprfpr_input": "models_train_test/car/tprfpr_car.csv",

    "output": "results/car_%s.csv",
    "out_model": "models_train_test/car/model_car.pkl",
    "out_train_dt": "models_train_test/car/train_data_car.csv",
    "out_test_dt": "models_train_test/car/test_data_car.csv",
    "out_scores": "models_train_test/car/scores_training_car.csv",
    "output_tprfpr": "models_train_test/car/tprfpr_car.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "churn": {
    "input": "datasets/churn.csv",
    "model_input": "models_train_test/churn/model_churn.pkl",
    "train_dt_input": "models_train_test/churn/train_data_churn.csv",
    "test_dt_input": "models_train_test/churn/test_data_churn.csv",
    "scores_input": "models_train_test/churn/scores_training_churn.csv",
    "tprfpr_input": "models_train_test/churn/tprfpr_churn.csv",

    "output": "results/churn_%s.csv",
    "out_model": "models_train_test/churn/model_churn.pkl",
    "out_train_dt": "models_train_test/churn/train_data_churn.csv",
    "out_test_dt": "models_train_test/churn/test_data_churn.csv",
    "out_scores": "models_train_test/churn/scores_training_churn.csv",
    "output_tprfpr": "models_train_test/churn/tprfpr_churn.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "clean2": {
    "input": "datasets/clean2.csv",
    "model_input": "models_train_test/clean2/model_clean2.pkl",
    "train_dt_input": "models_train_test/clean2/train_data_clean2.csv",
    "test_dt_input": "models_train_test/clean2/test_data_clean2.csv",
    "scores_input": "models_train_test/clean2/scores_training_clean2.csv",
    "tprfpr_input": "models_train_test/clean2/tprfpr_clean2.csv",

    "output": "results/clean2_%s.csv",
    "out_model": "models_train_test/clean2/model_clean2.pkl",
    "out_train_dt": "models_train_test/clean2/train_data_clean2.csv",
    "out_test_dt": "models_train_test/clean2/test_data_clean2.csv",
    "out_scores": "models_train_test/clean2/scores_training_clean2.csv",
    "output_tprfpr": "models_train_test/clean2/tprfpr_clean2.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "colleges_usnews": {
    "input": "datasets/colleges_usnews.csv",
    "model_input": "models_train_test/colleges_usnews/model_colleges_usnews.pkl",
    "train_dt_input": "models_train_test/colleges_usnews/train_data_colleges_usnews.csv",
    "test_dt_input": "models_train_test/colleges_usnews/test_data_colleges_usnews.csv",
    "scores_input": "models_train_test/colleges_usnews/scores_training_colleges_usnews.csv",
    "tprfpr_input": "models_train_test/colleges_usnews/tprfpr_colleges_usnews.csv",

    "output": "results/colleges_usnews_%s.csv",
    "out_model": "models_train_test/colleges_usnews/model_colleges_usnews.pkl",
    "out_train_dt": "models_train_test/colleges_usnews/train_data_colleges_usnews.csv",
    "out_test_dt": "models_train_test/colleges_usnews/test_data_colleges_usnews.csv",
    "out_scores": "models_train_test/colleges_usnews/scores_training_colleges_usnews.csv",
    "output_tprfpr": "models_train_test/colleges_usnews/tprfpr_colleges_usnews.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "compas_two_years": {
    "input": "datasets/compas_two_years.csv",
    "model_input": "models_train_test/compas_two_years/model_compas_two_years.pkl",
    "train_dt_input": "models_train_test/compas_two_years/train_data_compas_two_years.csv",
    "test_dt_input": "models_train_test/compas_two_years/test_data_compas_two_years.csv",
    "scores_input": "models_train_test/compas_two_years/scores_training_compas_two_years.csv",
    "tprfpr_input": "models_train_test/compas_two_years/tprfpr_compas_two_years.csv",

    "output": "results/compas_two_years_%s.csv",
    "out_model": "models_train_test/compas_two_years/model_compas_two_years.pkl",
    "out_train_dt": "models_train_test/compas_two_years/train_data_compas_two_years.csv",
    "out_test_dt": "models_train_test/compas_two_years/test_data_compas_two_years.csv",
    "out_scores": "models_train_test/compas_two_years/scores_training_compas_two_years.csv",
    "output_tprfpr": "models_train_test/compas_two_years/tprfpr_compas_two_years.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "cpu_act": {
    "input": "datasets/cpu_act.csv",
    "model_input": "models_train_test/cpu_act/model_cpu_act.pkl",
    "train_dt_input": "models_train_test/cpu_act/train_data_cpu_act.csv",
    "test_dt_input": "models_train_test/cpu_act/test_data_cpu_act.csv",
    "scores_input": "models_train_test/cpu_act/scores_training_cpu_act.csv",
    "tprfpr_input": "models_train_test/cpu_act/tprfpr_cpu_act.csv",

    "output": "results/cpu_act_%s.csv",
    "out_model": "models_train_test/cpu_act/model_cpu_act.pkl",
    "out_train_dt": "models_train_test/cpu_act/train_data_cpu_act.csv",
    "out_test_dt": "models_train_test/cpu_act/test_data_cpu_act.csv",
    "out_scores": "models_train_test/cpu_act/scores_training_cpu_act.csv",
    "output_tprfpr": "models_train_test/cpu_act/tprfpr_cpu_act.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "cpu_small": {
    "input": "datasets/cpu_small.csv",
    "model_input": "models_train_test/cpu_small/model_cpu_small.pkl",
    "train_dt_input": "models_train_test/cpu_small/train_data_cpu_small.csv",
    "test_dt_input": "models_train_test/cpu_small/test_data_cpu_small.csv",
    "scores_input": "models_train_test/cpu_small/scores_training_cpu_small.csv",
    "tprfpr_input": "models_train_test/cpu_small/tprfpr_cpu_small.csv",

    "output": "results/cpu_small_%s.csv",
    "out_model": "models_train_test/cpu_small/model_cpu_small.pkl",
    "out_train_dt": "models_train_test/cpu_small/train_data_cpu_small.csv",
    "out_test_dt": "models_train_test/cpu_small/test_data_cpu_small.csv",
    "out_scores": "models_train_test/cpu_small/scores_training_cpu_small.csv",
    "output_tprfpr": "models_train_test/cpu_small/tprfpr_cpu_small.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "delta_ailerons": {
    "input": "datasets/delta_ailerons.csv",
    "model_input": "models_train_test/delta_ailerons/model_delta_ailerons.pkl",
    "train_dt_input": "models_train_test/delta_ailerons/train_data_delta_ailerons.csv",
    "test_dt_input": "models_train_test/delta_ailerons/test_data_delta_ailerons.csv",
    "scores_input": "models_train_test/delta_ailerons/scores_training_delta_ailerons.csv",
    "tprfpr_input": "models_train_test/delta_ailerons/tprfpr_delta_ailerons.csv",

    "output": "results/delta_ailerons_%s.csv",
    "out_model": "models_train_test/delta_ailerons/model_delta_ailerons.pkl",
    "out_train_dt": "models_train_test/delta_ailerons/train_data_delta_ailerons.csv",
    "out_test_dt": "models_train_test/delta_ailerons/test_data_delta_ailerons.csv",
    "out_scores": "models_train_test/delta_ailerons/scores_training_delta_ailerons.csv",
    "output_tprfpr": "models_train_test/delta_ailerons/tprfpr_delta_ailerons.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "delta_elevators": {
    "input": "datasets/delta_elevators.csv",
    "model_input": "models_train_test/delta_elevators/model_delta_elevators.pkl",
    "train_dt_input": "models_train_test/delta_elevators/train_data_delta_elevators.csv",
    "test_dt_input": "models_train_test/delta_elevators/test_data_delta_elevators.csv",
    "scores_input": "models_train_test/delta_elevators/scores_training_delta_elevators.csv",
    "tprfpr_input": "models_train_test/delta_elevators/tprfpr_delta_elevators.csv",

    "output": "results/delta_elevators_%s.csv",
    "out_model": "models_train_test/delta_elevators/model_delta_elevators.pkl",
    "out_train_dt": "models_train_test/delta_elevators/train_data_delta_elevators.csv",
    "out_test_dt": "models_train_test/delta_elevators/test_data_delta_elevators.csv",
    "out_scores": "models_train_test/delta_elevators/scores_training_delta_elevators.csv",
    "output_tprfpr": "models_train_test/delta_elevators/tprfpr_delta_elevators.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "eeg_eye_state": {
    "input": "datasets/eeg_eye_state.csv",
    "model_input": "models_train_test/eeg_eye_state/model_eeg_eye_state.pkl",
    "train_dt_input": "models_train_test/eeg_eye_state/train_data_eeg_eye_state.csv",
    "test_dt_input": "models_train_test/eeg_eye_state/test_data_eeg_eye_state.csv",
    "scores_input": "models_train_test/eeg_eye_state/scores_training_eeg_eye_state.csv",
    "tprfpr_input": "models_train_test/eeg_eye_state/tprfpr_eeg_eye_state.csv",

    "output": "results/eeg_eye_state_%s.csv",
    "out_model": "models_train_test/eeg_eye_state/model_eeg_eye_state.pkl",
    "out_train_dt": "models_train_test/eeg_eye_state/train_data_eeg_eye_state.csv",
    "out_test_dt": "models_train_test/eeg_eye_state/test_data_eeg_eye_state.csv",
    "out_scores": "models_train_test/eeg_eye_state/scores_training_eeg_eye_state.csv",
    "output_tprfpr": "models_train_test/eeg_eye_state/tprfpr_eeg_eye_state.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "electricity": {
    "input": "datasets/electricity.csv",
    "model_input": "models_train_test/electricity/model_electricity.pkl",
    "train_dt_input": "models_train_test/electricity/train_data_electricity.csv",
    "test_dt_input": "models_train_test/electricity/test_data_electricity.csv",
    "scores_input": "models_train_test/electricity/scores_training_electricity.csv",
    "tprfpr_input": "models_train_test/electricity/tprfpr_electricity.csv",

    "output": "results/electricity_%s.csv",
    "out_model": "models_train_test/electricity/model_electricity.pkl",
    "out_train_dt": "models_train_test/electricity/train_data_electricity.csv",
    "out_test_dt": "models_train_test/electricity/test_data_electricity.csv",
    "out_scores": "models_train_test/electricity/scores_training_electricity.csv",
    "output_tprfpr": "models_train_test/electricity/tprfpr_electricity.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "elevators": {
    "input": "datasets/elevators.csv",
    "model_input": "models_train_test/elevators/model_elevators.pkl",
    "train_dt_input": "models_train_test/elevators/train_data_elevators.csv",
    "test_dt_input": "models_train_test/elevators/test_data_elevators.csv",
    "scores_input": "models_train_test/elevators/scores_training_elevators.csv",
    "tprfpr_input": "models_train_test/elevators/tprfpr_elevators.csv",

    "output": "results/elevators_%s.csv",
    "out_model": "models_train_test/elevators/model_elevators.pkl",
    "out_train_dt": "models_train_test/elevators/train_data_elevators.csv",
    "out_test_dt": "models_train_test/elevators/test_data_elevators.csv",
    "out_scores": "models_train_test/elevators/scores_training_elevators.csv",
    "output_tprfpr": "models_train_test/elevators/tprfpr_elevators.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c0_1000_5": {
    "input": "datasets/fri_c0_1000_5.csv",
    "model_input": "models_train_test/fri_c0_1000_5/model_fri_c0_1000_5.pkl",
    "train_dt_input": "models_train_test/fri_c0_1000_5/train_data_fri_c0_1000_5.csv",
    "test_dt_input": "models_train_test/fri_c0_1000_5/test_data_fri_c0_1000_5.csv",
    "scores_input": "models_train_test/fri_c0_1000_5/scores_training_fri_c0_1000_5.csv",
    "tprfpr_input": "models_train_test/fri_c0_1000_5/tprfpr_fri_c0_1000_5.csv",

    "output": "results/fri_c0_1000_5_%s.csv",
    "out_model": "models_train_test/fri_c0_1000_5/model_fri_c0_1000_5.pkl",
    "out_train_dt": "models_train_test/fri_c0_1000_5/train_data_fri_c0_1000_5.csv",
    "out_test_dt": "models_train_test/fri_c0_1000_5/test_data_fri_c0_1000_5.csv",
    "out_scores": "models_train_test/fri_c0_1000_5/scores_training_fri_c0_1000_5.csv",
    "output_tprfpr": "models_train_test/fri_c0_1000_5/tprfpr_fri_c0_1000_5.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c0_1000_10": {
    "input": "datasets/fri_c0_1000_10.csv",
    "model_input": "models_train_test/fri_c0_1000_10/model_fri_c0_1000_10.pkl",
    "train_dt_input": "models_train_test/fri_c0_1000_10/train_data_fri_c0_1000_10.csv",
    "test_dt_input": "models_train_test/fri_c0_1000_10/test_data_fri_c0_1000_10.csv",
    "scores_input": "models_train_test/fri_c0_1000_10/scores_training_fri_c0_1000_10.csv",
    "tprfpr_input": "models_train_test/fri_c0_1000_10/tprfpr_fri_c0_1000_10.csv",

    "output": "results/fri_c0_1000_10_%s.csv",
    "out_model": "models_train_test/fri_c0_1000_10/model_fri_c0_1000_10.pkl",
    "out_train_dt": "models_train_test/fri_c0_1000_10/train_data_fri_c0_1000_10.csv",
    "out_test_dt": "models_train_test/fri_c0_1000_10/test_data_fri_c0_1000_10.csv",
    "out_scores": "models_train_test/fri_c0_1000_10/scores_training_fri_c0_1000_10.csv",
    "output_tprfpr": "models_train_test/fri_c0_1000_10/tprfpr_fri_c0_1000_10.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c0_1000_25": {
    "input": "datasets/fri_c0_1000_25.csv",
    "model_input": "models_train_test/fri_c0_1000_25/model_fri_c0_1000_25.pkl",
    "train_dt_input": "models_train_test/fri_c0_1000_25/train_data_fri_c0_1000_25.csv",
    "test_dt_input": "models_train_test/fri_c0_1000_25/test_data_fri_c0_1000_25.csv",
    "scores_input": "models_train_test/fri_c0_1000_25/scores_training_fri_c0_1000_25.csv",
    "tprfpr_input": "models_train_test/fri_c0_1000_25/tprfpr_fri_c0_1000_25.csv",

    "output": "results/fri_c0_1000_25_%s.csv",
    "out_model": "models_train_test/fri_c0_1000_25/model_fri_c0_1000_25.pkl",
    "out_train_dt": "models_train_test/fri_c0_1000_25/train_data_fri_c0_1000_25.csv",
    "out_test_dt": "models_train_test/fri_c0_1000_25/test_data_fri_c0_1000_25.csv",
    "out_scores": "models_train_test/fri_c0_1000_25/scores_training_fri_c0_1000_25.csv",
    "output_tprfpr": "models_train_test/fri_c0_1000_25/tprfpr_fri_c0_1000_25.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c0_1000_50": {
    "input": "datasets/fri_c0_1000_50.csv",
    "model_input": "models_train_test/fri_c0_1000_50/model_fri_c0_1000_50.pkl",
    "train_dt_input": "models_train_test/fri_c0_1000_50/train_data_fri_c0_1000_50.csv",
    "test_dt_input": "models_train_test/fri_c0_1000_50/test_data_fri_c0_1000_50.csv",
    "scores_input": "models_train_test/fri_c0_1000_50/scores_training_fri_c0_1000_50.csv",
    "tprfpr_input": "models_train_test/fri_c0_1000_50/tprfpr_fri_c0_1000_50.csv",

    "output": "results/fri_c0_1000_50_%s.csv",
    "out_model": "models_train_test/fri_c0_1000_50/model_fri_c0_1000_50.pkl",
    "out_train_dt": "models_train_test/fri_c0_1000_50/train_data_fri_c0_1000_50.csv",
    "out_test_dt": "models_train_test/fri_c0_1000_50/test_data_fri_c0_1000_50.csv",
    "out_scores": "models_train_test/fri_c0_1000_50/scores_training_fri_c0_1000_50.csv",
    "output_tprfpr": "models_train_test/fri_c0_1000_50/tprfpr_fri_c0_1000_50.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c1_1000_5": {
    "input": "datasets/fri_c1_1000_5.csv",
    "model_input": "models_train_test/fri_c1_1000_5/model_fri_c1_1000_5.pkl",
    "train_dt_input": "models_train_test/fri_c1_1000_5/train_data_fri_c1_1000_5.csv",
    "test_dt_input": "models_train_test/fri_c1_1000_5/test_data_fri_c1_1000_5.csv",
    "scores_input": "models_train_test/fri_c1_1000_5/scores_training_fri_c1_1000_5.csv",
    "tprfpr_input": "models_train_test/fri_c1_1000_5/tprfpr_fri_c1_1000_5.csv",

    "output": "results/fri_c1_1000_5_%s.csv",
    "out_model": "models_train_test/fri_c1_1000_5/model_fri_c1_1000_5.pkl",
    "out_train_dt": "models_train_test/fri_c1_1000_5/train_data_fri_c1_1000_5.csv",
    "out_test_dt": "models_train_test/fri_c1_1000_5/test_data_fri_c1_1000_5.csv",
    "out_scores": "models_train_test/fri_c1_1000_5/scores_training_fri_c1_1000_5.csv",
    "output_tprfpr": "models_train_test/fri_c1_1000_5/tprfpr_fri_c1_1000_5.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c1_1000_10": {
    "input": "datasets/fri_c1_1000_10.csv",
    "model_input": "models_train_test/fri_c1_1000_10/model_fri_c1_1000_10.pkl",
    "train_dt_input": "models_train_test/fri_c1_1000_10/train_data_fri_c1_1000_10.csv",
    "test_dt_input": "models_train_test/fri_c1_1000_10/test_data_fri_c1_1000_10.csv",
    "scores_input": "models_train_test/fri_c1_1000_10/scores_training_fri_c1_1000_10.csv",
    "tprfpr_input": "models_train_test/fri_c1_1000_10/tprfpr_fri_c1_1000_10.csv",

    "output": "results/fri_c1_1000_10_%s.csv",
    "out_model": "models_train_test/fri_c1_1000_10/model_fri_c1_1000_10.pkl",
    "out_train_dt": "models_train_test/fri_c1_1000_10/train_data_fri_c1_1000_10.csv",
    "out_test_dt": "models_train_test/fri_c1_1000_10/test_data_fri_c1_1000_10.csv",
    "out_scores": "models_train_test/fri_c1_1000_10/scores_training_fri_c1_1000_10.csv",
    "output_tprfpr": "models_train_test/fri_c1_1000_10/tprfpr_fri_c1_1000_10.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c1_1000_25": {
    "input": "datasets/fri_c1_1000_25.csv",
    "model_input": "models_train_test/fri_c1_1000_25/model_fri_c1_1000_25.pkl",
    "train_dt_input": "models_train_test/fri_c1_1000_25/train_data_fri_c1_1000_25.csv",
    "test_dt_input": "models_train_test/fri_c1_1000_25/test_data_fri_c1_1000_25.csv",
    "scores_input": "models_train_test/fri_c1_1000_25/scores_training_fri_c1_1000_25.csv",
    "tprfpr_input": "models_train_test/fri_c1_1000_25/tprfpr_fri_c1_1000_25.csv",

    "output": "results/fri_c1_1000_25_%s.csv",
    "out_model": "models_train_test/fri_c1_1000_25/model_fri_c1_1000_25.pkl",
    "out_train_dt": "models_train_test/fri_c1_1000_25/train_data_fri_c1_1000_25.csv",
    "out_test_dt": "models_train_test/fri_c1_1000_25/test_data_fri_c1_1000_25.csv",
    "out_scores": "models_train_test/fri_c1_1000_25/scores_training_fri_c1_1000_25.csv",
    "output_tprfpr": "models_train_test/fri_c1_1000_25/tprfpr_fri_c1_1000_25.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c1_1000_50": {
    "input": "datasets/fri_c1_1000_50.csv",
    "model_input": "models_train_test/fri_c1_1000_50/model_fri_c1_1000_50.pkl",
    "train_dt_input": "models_train_test/fri_c1_1000_50/train_data_fri_c1_1000_50.csv",
    "test_dt_input": "models_train_test/fri_c1_1000_50/test_data_fri_c1_1000_50.csv",
    "scores_input": "models_train_test/fri_c1_1000_50/scores_training_fri_c1_1000_50.csv",
    "tprfpr_input": "models_train_test/fri_c1_1000_50/tprfpr_fri_c1_1000_50.csv",

    "output": "results/fri_c1_1000_50_%s.csv",
    "out_model": "models_train_test/fri_c1_1000_50/model_fri_c1_1000_50.pkl",
    "out_train_dt": "models_train_test/fri_c1_1000_50/train_data_fri_c1_1000_50.csv",
    "out_test_dt": "models_train_test/fri_c1_1000_50/test_data_fri_c1_1000_50.csv",
    "out_scores": "models_train_test/fri_c1_1000_50/scores_training_fri_c1_1000_50.csv",
    "output_tprfpr": "models_train_test/fri_c1_1000_50/tprfpr_fri_c1_1000_50.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c2_1000_5": {
    "input": "datasets/fri_c2_1000_5.csv",
    "model_input": "models_train_test/fri_c2_1000_5/model_fri_c2_1000_5.pkl",
    "train_dt_input": "models_train_test/fri_c2_1000_5/train_data_fri_c2_1000_5.csv",
    "test_dt_input": "models_train_test/fri_c2_1000_5/test_data_fri_c2_1000_5.csv",
    "scores_input": "models_train_test/fri_c2_1000_5/scores_training_fri_c2_1000_5.csv",
    "tprfpr_input": "models_train_test/fri_c2_1000_5/tprfpr_fri_c2_1000_5.csv",

    "output": "results/fri_c2_1000_5_%s.csv",
    "out_model": "models_train_test/fri_c2_1000_5/model_fri_c2_1000_5.pkl",
    "out_train_dt": "models_train_test/fri_c2_1000_5/train_data_fri_c2_1000_5.csv",
    "out_test_dt": "models_train_test/fri_c2_1000_5/test_data_fri_c2_1000_5.csv",
    "out_scores": "models_train_test/fri_c2_1000_5/scores_training_fri_c2_1000_5.csv",
    "output_tprfpr": "models_train_test/fri_c2_1000_5/tprfpr_fri_c2_1000_5.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c2_1000_10": {
    "input": "datasets/fri_c2_1000_10.csv",
    "model_input": "models_train_test/fri_c2_1000_10/model_fri_c2_1000_10.pkl",
    "train_dt_input": "models_train_test/fri_c2_1000_10/train_data_fri_c2_1000_10.csv",
    "test_dt_input": "models_train_test/fri_c2_1000_10/test_data_fri_c2_1000_10.csv",
    "scores_input": "models_train_test/fri_c2_1000_10/scores_training_fri_c2_1000_10.csv",
    "tprfpr_input": "models_train_test/fri_c2_1000_10/tprfpr_fri_c2_1000_10.csv",

    "output": "results/fri_c2_1000_10_%s.csv",
    "out_model": "models_train_test/fri_c2_1000_10/model_fri_c2_1000_10.pkl",
    "out_train_dt": "models_train_test/fri_c2_1000_10/train_data_fri_c2_1000_10.csv",
    "out_test_dt": "models_train_test/fri_c2_1000_10/test_data_fri_c2_1000_10.csv",
    "out_scores": "models_train_test/fri_c2_1000_10/scores_training_fri_c2_1000_10.csv",
    "output_tprfpr": "models_train_test/fri_c2_1000_10/tprfpr_fri_c2_1000_10.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c2_1000_25": {
    "input": "datasets/fri_c2_1000_25.csv",
    "model_input": "models_train_test/fri_c2_1000_25/model_fri_c2_1000_25.pkl",
    "train_dt_input": "models_train_test/fri_c2_1000_25/train_data_fri_c2_1000_25.csv",
    "test_dt_input": "models_train_test/fri_c2_1000_25/test_data_fri_c2_1000_25.csv",
    "scores_input": "models_train_test/fri_c2_1000_25/scores_training_fri_c2_1000_25.csv",
    "tprfpr_input": "models_train_test/fri_c2_1000_25/tprfpr_fri_c2_1000_25.csv",

    "output": "results/fri_c2_1000_25_%s.csv",
    "out_model": "models_train_test/fri_c2_1000_25/model_fri_c2_1000_25.pkl",
    "out_train_dt": "models_train_test/fri_c2_1000_25/train_data_fri_c2_1000_25.csv",
    "out_test_dt": "models_train_test/fri_c2_1000_25/test_data_fri_c2_1000_25.csv",
    "out_scores": "models_train_test/fri_c2_1000_25/scores_training_fri_c2_1000_25.csv",
    "output_tprfpr": "models_train_test/fri_c2_1000_25/tprfpr_fri_c2_1000_25.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c2_1000_50": {
    "input": "datasets/fri_c2_1000_50.csv",
    "model_input": "models_train_test/fri_c2_1000_50/model_fri_c2_1000_50.pkl",
    "train_dt_input": "models_train_test/fri_c2_1000_50/train_data_fri_c2_1000_50.csv",
    "test_dt_input": "models_train_test/fri_c2_1000_50/test_data_fri_c2_1000_50.csv",
    "scores_input": "models_train_test/fri_c2_1000_50/scores_training_fri_c2_1000_50.csv",
    "tprfpr_input": "models_train_test/fri_c2_1000_50/tprfpr_fri_c2_1000_50.csv",

    "output": "results/fri_c2_1000_50_%s.csv",
    "out_model": "models_train_test/fri_c2_1000_50/model_fri_c2_1000_50.pkl",
    "out_train_dt": "models_train_test/fri_c2_1000_50/train_data_fri_c2_1000_50.csv",
    "out_test_dt": "models_train_test/fri_c2_1000_50/test_data_fri_c2_1000_50.csv",
    "out_scores": "models_train_test/fri_c2_1000_50/scores_training_fri_c2_1000_50.csv",
    "output_tprfpr": "models_train_test/fri_c2_1000_50/tprfpr_fri_c2_1000_50.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c3_1000_5": {
    "input": "datasets/fri_c3_1000_5.csv",
    "model_input": "models_train_test/fri_c3_1000_5/model_fri_c3_1000_5.pkl",
    "train_dt_input": "models_train_test/fri_c3_1000_5/train_data_fri_c3_1000_5.csv",
    "test_dt_input": "models_train_test/fri_c3_1000_5/test_data_fri_c3_1000_5.csv",
    "scores_input": "models_train_test/fri_c3_1000_5/scores_training_fri_c3_1000_5.csv",
    "tprfpr_input": "models_train_test/fri_c3_1000_5/tprfpr_fri_c3_1000_5.csv",

    "output": "results/fri_c3_1000_5_%s.csv",
    "out_model": "models_train_test/fri_c3_1000_5/model_fri_c3_1000_5.pkl",
    "out_train_dt": "models_train_test/fri_c3_1000_5/train_data_fri_c3_1000_5.csv",
    "out_test_dt": "models_train_test/fri_c3_1000_5/test_data_fri_c3_1000_5.csv",
    "out_scores": "models_train_test/fri_c3_1000_5/scores_training_fri_c3_1000_5.csv",
    "output_tprfpr": "models_train_test/fri_c3_1000_5/tprfpr_fri_c3_1000_5.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c3_1000_10": {
    "input": "datasets/fri_c3_1000_10.csv",
    "model_input": "models_train_test/fri_c3_1000_10/model_fri_c3_1000_10.pkl",
    "train_dt_input": "models_train_test/fri_c3_1000_10/train_data_fri_c3_1000_10.csv",
    "test_dt_input": "models_train_test/fri_c3_1000_10/test_data_fri_c3_1000_10.csv",
    "scores_input": "models_train_test/fri_c3_1000_10/scores_training_fri_c3_1000_10.csv",
    "tprfpr_input": "models_train_test/fri_c3_1000_10/tprfpr_fri_c3_1000_10.csv",

    "output": "results/fri_c3_1000_10_%s.csv",
    "out_model": "models_train_test/fri_c3_1000_10/model_fri_c3_1000_10.pkl",
    "out_train_dt": "models_train_test/fri_c3_1000_10/train_data_fri_c3_1000_10.csv",
    "out_test_dt": "models_train_test/fri_c3_1000_10/test_data_fri_c3_1000_10.csv",
    "out_scores": "models_train_test/fri_c3_1000_10/scores_training_fri_c3_1000_10.csv",
    "output_tprfpr": "models_train_test/fri_c3_1000_10/tprfpr_fri_c3_1000_10.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c3_1000_25": {
    "input": "datasets/fri_c3_1000_25.csv",
    "model_input": "models_train_test/fri_c3_1000_25/model_fri_c3_1000_25.pkl",
    "train_dt_input": "models_train_test/fri_c3_1000_25/train_data_fri_c3_1000_25.csv",
    "test_dt_input": "models_train_test/fri_c3_1000_25/test_data_fri_c3_1000_25.csv",
    "scores_input": "models_train_test/fri_c3_1000_25/scores_training_fri_c3_1000_25.csv",
    "tprfpr_input": "models_train_test/fri_c3_1000_25/tprfpr_fri_c3_1000_25.csv",

    "output": "results/fri_c3_1000_25_%s.csv",
    "out_model": "models_train_test/fri_c3_1000_25/model_fri_c3_1000_25.pkl",
    "out_train_dt": "models_train_test/fri_c3_1000_25/train_data_fri_c3_1000_25.csv",
    "out_test_dt": "models_train_test/fri_c3_1000_25/test_data_fri_c3_1000_25.csv",
    "out_scores": "models_train_test/fri_c3_1000_25/scores_training_fri_c3_1000_25.csv",
    "output_tprfpr": "models_train_test/fri_c3_1000_25/tprfpr_fri_c3_1000_25.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c3_1000_50": {
    "input": "datasets/fri_c3_1000_50.csv",
    "model_input": "models_train_test/fri_c3_1000_50/model_fri_c3_1000_50.pkl",
    "train_dt_input": "models_train_test/fri_c3_1000_50/train_data_fri_c3_1000_50.csv",
    "test_dt_input": "models_train_test/fri_c3_1000_50/test_data_fri_c3_1000_50.csv",
    "scores_input": "models_train_test/fri_c3_1000_50/scores_training_fri_c3_1000_50.csv",
    "tprfpr_input": "models_train_test/fri_c3_1000_50/tprfpr_fri_c3_1000_50.csv",

    "output": "results/fri_c3_1000_50_%s.csv",
    "out_model": "models_train_test/fri_c3_1000_50/model_fri_c3_1000_50.pkl",
    "out_train_dt": "models_train_test/fri_c3_1000_50/train_data_fri_c3_1000_50.csv",
    "out_test_dt": "models_train_test/fri_c3_1000_50/test_data_fri_c3_1000_50.csv",
    "out_scores": "models_train_test/fri_c3_1000_50/scores_training_fri_c3_1000_50.csv",
    "output_tprfpr": "models_train_test/fri_c3_1000_50/tprfpr_fri_c3_1000_50.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c4_1000_10": {
    "input": "datasets/fri_c4_1000_10.csv",
    "model_input": "models_train_test/fri_c4_1000_10/model_fri_c4_1000_10.pkl",
    "train_dt_input": "models_train_test/fri_c4_1000_10/train_data_fri_c4_1000_10.csv",
    "test_dt_input": "models_train_test/fri_c4_1000_10/test_data_fri_c4_1000_10.csv",
    "scores_input": "models_train_test/fri_c4_1000_10/scores_training_fri_c4_1000_10.csv",
    "tprfpr_input": "models_train_test/fri_c4_1000_10/tprfpr_fri_c4_1000_10.csv",

    "output": "results/fri_c4_1000_10_%s.csv",
    "out_model": "models_train_test/fri_c4_1000_10/model_fri_c4_1000_10.pkl",
    "out_train_dt": "models_train_test/fri_c4_1000_10/train_data_fri_c4_1000_10.csv",
    "out_test_dt": "models_train_test/fri_c4_1000_10/test_data_fri_c4_1000_10.csv",
    "out_scores": "models_train_test/fri_c4_1000_10/scores_training_fri_c4_1000_10.csv",
    "output_tprfpr": "models_train_test/fri_c4_1000_10/tprfpr_fri_c4_1000_10.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c4_1000_25": {
    "input": "datasets/fri_c4_1000_25.csv",
    "model_input": "models_train_test/fri_c4_1000_25/model_fri_c4_1000_25.pkl",
    "train_dt_input": "models_train_test/fri_c4_1000_25/train_data_fri_c4_1000_25.csv",
    "test_dt_input": "models_train_test/fri_c4_1000_25/test_data_fri_c4_1000_25.csv",
    "scores_input": "models_train_test/fri_c4_1000_25/scores_training_fri_c4_1000_25.csv",
    "tprfpr_input": "models_train_test/fri_c4_1000_25/tprfpr_fri_c4_1000_25.csv",

    "output": "results/fri_c4_1000_25_%s.csv",
    "out_model": "models_train_test/fri_c4_1000_25/model_fri_c4_1000_25.pkl",
    "out_train_dt": "models_train_test/fri_c4_1000_25/train_data_fri_c4_1000_25.csv",
    "out_test_dt": "models_train_test/fri_c4_1000_25/test_data_fri_c4_1000_25.csv",
    "out_scores": "models_train_test/fri_c4_1000_25/scores_training_fri_c4_1000_25.csv",
    "output_tprfpr": "models_train_test/fri_c4_1000_25/tprfpr_fri_c4_1000_25.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c4_1000_50": {
    "input": "datasets/fri_c4_1000_50.csv",
    "model_input": "models_train_test/fri_c4_1000_50/model_fri_c4_1000_50.pkl",
    "train_dt_input": "models_train_test/fri_c4_1000_50/train_data_fri_c4_1000_50.csv",
    "test_dt_input": "models_train_test/fri_c4_1000_50/test_data_fri_c4_1000_50.csv",
    "scores_input": "models_train_test/fri_c4_1000_50/scores_training_fri_c4_1000_50.csv",
    "tprfpr_input": "models_train_test/fri_c4_1000_50/tprfpr_fri_c4_1000_50.csv",

    "output": "results/fri_c4_1000_50_%s.csv",
    "out_model": "models_train_test/fri_c4_1000_50/model_fri_c4_1000_50.pkl",
    "out_train_dt": "models_train_test/fri_c4_1000_50/train_data_fri_c4_1000_50.csv",
    "out_test_dt": "models_train_test/fri_c4_1000_50/test_data_fri_c4_1000_50.csv",
    "out_scores": "models_train_test/fri_c4_1000_50/scores_training_fri_c4_1000_50.csv",
    "output_tprfpr": "models_train_test/fri_c4_1000_50/tprfpr_fri_c4_1000_50.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fri_c4_1000_100": {
    "input": "datasets/fri_c4_1000_100.csv",
    "model_input": "models_train_test/fri_c4_1000_100/model_fri_c4_1000_100.pkl",
    "train_dt_input": "models_train_test/fri_c4_1000_100/train_data_fri_c4_1000_100.csv",
    "test_dt_input": "models_train_test/fri_c4_1000_100/test_data_fri_c4_1000_100.csv",
    "scores_input": "models_train_test/fri_c4_1000_100/scores_training_fri_c4_1000_100.csv",
    "tprfpr_input": "models_train_test/fri_c4_1000_100/tprfpr_fri_c4_1000_100.csv",

    "output": "results/fri_c4_1000_100_%s.csv",
    "out_model": "models_train_test/fri_c4_1000_100/model_fri_c4_1000_100.pkl",
    "out_train_dt": "models_train_test/fri_c4_1000_100/train_data_fri_c4_1000_100.csv",
    "out_test_dt": "models_train_test/fri_c4_1000_100/test_data_fri_c4_1000_100.csv",
    "out_scores": "models_train_test/fri_c4_1000_100/scores_training_fri_c4_1000_100.csv",
    "output_tprfpr": "models_train_test/fri_c4_1000_100/tprfpr_fri_c4_1000_100.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "fried": {
    "input": "datasets/fried.csv",
    "model_input": "models_train_test/fried/model_fried.pkl",
    "train_dt_input": "models_train_test/fried/train_data_fried.csv",
    "test_dt_input": "models_train_test/fried/test_data_fried.csv",
    "scores_input": "models_train_test/fried/scores_training_fried.csv",
    "tprfpr_input": "models_train_test/fried/tprfpr_fried.csv",

    "output": "results/fried_%s.csv",
    "out_model": "models_train_test/fried/model_fried.pkl",
    "out_train_dt": "models_train_test/fried/train_data_fried.csv",
    "out_test_dt": "models_train_test/fried/test_data_fried.csv",
    "out_scores": "models_train_test/fried/scores_training_fried.csv",
    "output_tprfpr": "models_train_test/fried/tprfpr_fried.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1": {
    "input": "datasets/GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1.csv",
    "model_input": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1/model_GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1.pkl",
    "train_dt_input": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1/train_data_GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1.csv",
    "test_dt_input": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1/test_data_GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1.csv",
    "scores_input": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1/scores_training_GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1.csv",
    "tprfpr_input": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1/tprfpr_GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1.csv",

    "output": "results/GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1_%s.csv",
    "out_model": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1/model_GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1.pkl",
    "out_train_dt": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1/train_data_GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1.csv",
    "out_test_dt": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1/test_data_GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1.csv",
    "out_scores": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1/scores_training_GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1.csv",
    "output_tprfpr": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1/tprfpr_GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1": {
    "input": "datasets/GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1.csv",
    "model_input": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1/model_GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1.pkl",
    "train_dt_input": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1/train_data_GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1.csv",
    "test_dt_input": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1/test_data_GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1.csv",
    "scores_input": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1/scores_training_GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1.csv",
    "tprfpr_input": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1/tprfpr_GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1.csv",

    "output": "results/GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1_%s.csv",
    "out_model": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1/model_GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1.pkl",
    "out_train_dt": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1/train_data_GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1.csv",
    "out_test_dt": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1/test_data_GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1.csv",
    "out_scores": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1/scores_training_GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1.csv",
    "output_tprfpr": "models_train_test/GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1/tprfpr_GAMETES_Epistasis_2_Way_20atts_0_4H_EDM_1_1.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1": {
    "input": "datasets/GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1.csv",
    "model_input": "models_train_test/GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1/model_GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1.pkl",
    "train_dt_input": "models_train_test/GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1/train_data_GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1.csv",
    "test_dt_input": "models_train_test/GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1/test_data_GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1.csv",
    "scores_input": "models_train_test/GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1/scores_training_GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1.csv",
    "tprfpr_input": "models_train_test/GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1/tprfpr_GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1.csv",

    "output": "results/GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1_%s.csv",
    "out_model": "models_train_test/GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1/model_GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1.pkl",
    "out_train_dt": "models_train_test/GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1/train_data_GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1.csv",
    "out_test_dt": "models_train_test/GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1/test_data_GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1.csv",
    "out_scores": "models_train_test/GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1/scores_training_GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1.csv",
    "output_tprfpr": "models_train_test/GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1/tprfpr_GAMETES_Epistasis_3_Way_20atts_0_2H_EDM_1_1.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001": {
    "input": "datasets/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001.csv",
    "model_input": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001/model_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001.pkl",
    "train_dt_input": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001/train_data_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001.csv",
    "test_dt_input": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001/test_data_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001.csv",
    "scores_input": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001/scores_training_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001.csv",
    "tprfpr_input": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001/tprfpr_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001.csv",

    "output": "results/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001_%s.csv",
    "out_model": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001/model_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001.pkl",
    "out_train_dt": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001/train_data_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001.csv",
    "out_test_dt": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001/test_data_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001.csv",
    "out_scores": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001/scores_training_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001.csv",
    "output_tprfpr": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001/tprfpr_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM_001.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001": {
    "input": "datasets/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001.csv",
    "model_input": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001/model_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001.pkl",
    "train_dt_input": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001/train_data_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001.csv",
    "test_dt_input": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001/test_data_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001.csv",
    "scores_input": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001/scores_training_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001.csv",
    "tprfpr_input": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001/tprfpr_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001.csv",

    "output": "results/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001_%s.csv",
    "out_model": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001/model_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001.pkl",
    "out_train_dt": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001/train_data_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001.csv",
    "out_test_dt": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001/test_data_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001.csv",
    "out_scores": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001/scores_training_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001.csv",
    "output_tprfpr": "models_train_test/GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001/tprfpr_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM_2_001.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "higgs": {
    "input": "datasets/higgs.csv",
    "model_input": "models_train_test/higgs/model_higgs.pkl",
    "train_dt_input": "models_train_test/higgs/train_data_higgs.csv",
    "test_dt_input": "models_train_test/higgs/test_data_higgs.csv",
    "scores_input": "models_train_test/higgs/scores_training_higgs.csv",
    "tprfpr_input": "models_train_test/higgs/tprfpr_higgs.csv",

    "output": "results/higgs_%s.csv",
    "out_model": "models_train_test/higgs/model_higgs.pkl",
    "out_train_dt": "models_train_test/higgs/train_data_higgs.csv",
    "out_test_dt": "models_train_test/higgs/test_data_higgs.csv",
    "out_scores": "models_train_test/higgs/scores_training_higgs.csv",
    "output_tprfpr": "models_train_test/higgs/tprfpr_higgs.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "hill_valley": {
    "input": "datasets/hill_valley.csv",
    "model_input": "models_train_test/hill_valley/model_hill_valley.pkl",
    "train_dt_input": "models_train_test/hill_valley/train_data_hill_valley.csv",
    "test_dt_input": "models_train_test/hill_valley/test_data_hill_valley.csv",
    "scores_input": "models_train_test/hill_valley/scores_training_hill_valley.csv",
    "tprfpr_input": "models_train_test/hill_valley/tprfpr_hill_valley.csv",

    "output": "results/hill_valley_%s.csv",
    "out_model": "models_train_test/hill_valley/model_hill_valley.pkl",
    "out_train_dt": "models_train_test/hill_valley/train_data_hill_valley.csv",
    "out_test_dt": "models_train_test/hill_valley/test_data_hill_valley.csv",
    "out_scores": "models_train_test/hill_valley/scores_training_hill_valley.csv",
    "output_tprfpr": "models_train_test/hill_valley/tprfpr_hill_valley.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "hill_valley1566": {
    "input": "datasets/hill_valley1566.csv",
    "model_input": "models_train_test/hill_valley1566/model_hill_valley1566.pkl",
    "train_dt_input": "models_train_test/hill_valley1566/train_data_hill_valley1566.csv",
    "test_dt_input": "models_train_test/hill_valley1566/test_data_hill_valley1566.csv",
    "scores_input": "models_train_test/hill_valley1566/scores_training_hill_valley1566.csv",
    "tprfpr_input": "models_train_test/hill_valley1566/tprfpr_hill_valley1566.csv",

    "output": "results/hill_valley1566_%s.csv",
    "out_model": "models_train_test/hill_valley1566/model_hill_valley1566.pkl",
    "out_train_dt": "models_train_test/hill_valley1566/train_data_hill_valley1566.csv",
    "out_test_dt": "models_train_test/hill_valley1566/test_data_hill_valley1566.csv",
    "out_scores": "models_train_test/hill_valley1566/scores_training_hill_valley1566.csv",
    "output_tprfpr": "models_train_test/hill_valley1566/tprfpr_hill_valley1566.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "house_8L": {
    "input": "datasets/house_8L.csv",
    "model_input": "models_train_test/house_8L/model_house_8L.pkl",
    "train_dt_input": "models_train_test/house_8L/train_data_house_8L.csv",
    "test_dt_input": "models_train_test/house_8L/test_data_house_8L.csv",
    "scores_input": "models_train_test/house_8L/scores_training_house_8L.csv",
    "tprfpr_input": "models_train_test/house_8L/tprfpr_house_8L.csv",

    "output": "results/house_8L_%s.csv",
    "out_model": "models_train_test/house_8L/model_house_8L.pkl",
    "out_train_dt": "models_train_test/house_8L/train_data_house_8L.csv",
    "out_test_dt": "models_train_test/house_8L/test_data_house_8L.csv",
    "out_scores": "models_train_test/house_8L/scores_training_house_8L.csv",
    "output_tprfpr": "models_train_test/house_8L/tprfpr_house_8L.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "house_16H": {
    "input": "datasets/house_16H.csv",
    "model_input": "models_train_test/house_16H/model_house_16H.pkl",
    "train_dt_input": "models_train_test/house_16H/train_data_house_16H.csv",
    "test_dt_input": "models_train_test/house_16H/test_data_house_16H.csv",
    "scores_input": "models_train_test/house_16H/scores_training_house_16H.csv",
    "tprfpr_input": "models_train_test/house_16H/tprfpr_house_16H.csv",

    "output": "results/house_16H_%s.csv",
    "out_model": "models_train_test/house_16H/model_house_16H.pkl",
    "out_train_dt": "models_train_test/house_16H/train_data_house_16H.csv",
    "out_test_dt": "models_train_test/house_16H/test_data_house_16H.csv",
    "out_scores": "models_train_test/house_16H/scores_training_house_16H.csv",
    "output_tprfpr": "models_train_test/house_16H/tprfpr_house_16H.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "houses": {
    "input": "datasets/houses.csv",
    "model_input": "models_train_test/houses/model_houses.pkl",
    "train_dt_input": "models_train_test/houses/train_data_houses.csv",
    "test_dt_input": "models_train_test/houses/test_data_houses.csv",
    "scores_input": "models_train_test/houses/scores_training_houses.csv",
    "tprfpr_input": "models_train_test/houses/tprfpr_houses.csv",

    "output": "results/houses_%s.csv",
    "out_model": "models_train_test/houses/model_houses.pkl",
    "out_train_dt": "models_train_test/houses/train_data_houses.csv",
    "out_test_dt": "models_train_test/houses/test_data_houses.csv",
    "out_scores": "models_train_test/houses/scores_training_houses.csv",
    "output_tprfpr": "models_train_test/houses/tprfpr_houses.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "ipums_la_98_small": {
    "input": "datasets/ipums_la_98_small.csv",
    "model_input": "models_train_test/ipums_la_98_small/model_ipums_la_98_small.pkl",
    "train_dt_input": "models_train_test/ipums_la_98_small/train_data_ipums_la_98_small.csv",
    "test_dt_input": "models_train_test/ipums_la_98_small/test_data_ipums_la_98_small.csv",
    "scores_input": "models_train_test/ipums_la_98_small/scores_training_ipums_la_98_small.csv",
    "tprfpr_input": "models_train_test/ipums_la_98_small/tprfpr_ipums_la_98_small.csv",

    "output": "results/ipums_la_98_small_%s.csv",
    "out_model": "models_train_test/ipums_la_98_small/model_ipums_la_98_small.pkl",
    "out_train_dt": "models_train_test/ipums_la_98_small/train_data_ipums_la_98_small.csv",
    "out_test_dt": "models_train_test/ipums_la_98_small/test_data_ipums_la_98_small.csv",
    "out_scores": "models_train_test/ipums_la_98_small/scores_training_ipums_la_98_small.csv",
    "output_tprfpr": "models_train_test/ipums_la_98_small/tprfpr_ipums_la_98_small.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "ipums_la_99_small": {
    "input": "datasets/ipums_la_99_small.csv",
    "model_input": "models_train_test/ipums_la_99_small/model_ipums_la_99_small.pkl",
    "train_dt_input": "models_train_test/ipums_la_99_small/train_data_ipums_la_99_small.csv",
    "test_dt_input": "models_train_test/ipums_la_99_small/test_data_ipums_la_99_small.csv",
    "scores_input": "models_train_test/ipums_la_99_small/scores_training_ipums_la_99_small.csv",
    "tprfpr_input": "models_train_test/ipums_la_99_small/tprfpr_ipums_la_99_small.csv",

    "output": "results/ipums_la_99_small_%s.csv",
    "out_model": "models_train_test/ipums_la_99_small/model_ipums_la_99_small.pkl",
    "out_train_dt": "models_train_test/ipums_la_99_small/train_data_ipums_la_99_small.csv",
    "out_test_dt": "models_train_test/ipums_la_99_small/test_data_ipums_la_99_small.csv",
    "out_scores": "models_train_test/ipums_la_99_small/scores_training_ipums_la_99_small.csv",
    "output_tprfpr": "models_train_test/ipums_la_99_small/tprfpr_ipums_la_99_small.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "JapaneseVowels": {
    "input": "datasets/JapaneseVowels.csv",
    "model_input": "models_train_test/JapaneseVowels/model_JapaneseVowels.pkl",
    "train_dt_input": "models_train_test/JapaneseVowels/train_data_JapaneseVowels.csv",
    "test_dt_input": "models_train_test/JapaneseVowels/test_data_JapaneseVowels.csv",
    "scores_input": "models_train_test/JapaneseVowels/scores_training_JapaneseVowels.csv",
    "tprfpr_input": "models_train_test/JapaneseVowels/tprfpr_JapaneseVowels.csv",

    "output": "results/JapaneseVowels_%s.csv",
    "out_model": "models_train_test/JapaneseVowels/model_JapaneseVowels.pkl",
    "out_train_dt": "models_train_test/JapaneseVowels/train_data_JapaneseVowels.csv",
    "out_test_dt": "models_train_test/JapaneseVowels/test_data_JapaneseVowels.csv",
    "out_scores": "models_train_test/JapaneseVowels/scores_training_JapaneseVowels.csv",
    "output_tprfpr": "models_train_test/JapaneseVowels/tprfpr_JapaneseVowels.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "jasmine": {
    "input": "datasets/jasmine.csv",
    "model_input": "models_train_test/jasmine/model_jasmine.pkl",
    "train_dt_input": "models_train_test/jasmine/train_data_jasmine.csv",
    "test_dt_input": "models_train_test/jasmine/test_data_jasmine.csv",
    "scores_input": "models_train_test/jasmine/scores_training_jasmine.csv",
    "tprfpr_input": "models_train_test/jasmine/tprfpr_jasmine.csv",

    "output": "results/jasmine_%s.csv",
    "out_model": "models_train_test/jasmine/model_jasmine.pkl",
    "out_train_dt": "models_train_test/jasmine/train_data_jasmine.csv",
    "out_test_dt": "models_train_test/jasmine/test_data_jasmine.csv",
    "out_scores": "models_train_test/jasmine/scores_training_jasmine.csv",
    "output_tprfpr": "models_train_test/jasmine/tprfpr_jasmine.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "jungle_elephant": {
    "input": "datasets/jungle_elephant.csv",
    "model_input": "models_train_test/jungle_elephant/model_jungle_elephant.pkl",
    "train_dt_input": "models_train_test/jungle_elephant/train_data_jungle_elephant.csv",
    "test_dt_input": "models_train_test/jungle_elephant/test_data_jungle_elephant.csv",
    "scores_input": "models_train_test/jungle_elephant/scores_training_jungle_elephant.csv",
    "tprfpr_input": "models_train_test/jungle_elephant/tprfpr_jungle_elephant.csv",

    "output": "results/jungle_elephant_%s.csv",
    "out_model": "models_train_test/jungle_elephant/model_jungle_elephant.pkl",
    "out_train_dt": "models_train_test/jungle_elephant/train_data_jungle_elephant.csv",
    "out_test_dt": "models_train_test/jungle_elephant/test_data_jungle_elephant.csv",
    "out_scores": "models_train_test/jungle_elephant/scores_training_jungle_elephant.csv",
    "output_tprfpr": "models_train_test/jungle_elephant/tprfpr_jungle_elephant.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "jungle_lion": {
    "input": "datasets/jungle_lion.csv",
    "model_input": "models_train_test/jungle_lion/model_jungle_lion.pkl",
    "train_dt_input": "models_train_test/jungle_lion/train_data_jungle_lion.csv",
    "test_dt_input": "models_train_test/jungle_lion/test_data_jungle_lion.csv",
    "scores_input": "models_train_test/jungle_lion/scores_training_jungle_lion.csv",
    "tprfpr_input": "models_train_test/jungle_lion/tprfpr_jungle_lion.csv",

    "output": "results/jungle_lion_%s.csv",
    "out_model": "models_train_test/jungle_lion/model_jungle_lion.pkl",
    "out_train_dt": "models_train_test/jungle_lion/train_data_jungle_lion.csv",
    "out_test_dt": "models_train_test/jungle_lion/test_data_jungle_lion.csv",
    "out_scores": "models_train_test/jungle_lion/scores_training_jungle_lion.csv",
    "output_tprfpr": "models_train_test/jungle_lion/tprfpr_jungle_lion.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "jungle_rat": {
    "input": "datasets/jungle_rat.csv",
    "model_input": "models_train_test/jungle_rat/model_jungle_rat.pkl",
    "train_dt_input": "models_train_test/jungle_rat/train_data_jungle_rat.csv",
    "test_dt_input": "models_train_test/jungle_rat/test_data_jungle_rat.csv",
    "scores_input": "models_train_test/jungle_rat/scores_training_jungle_rat.csv",
    "tprfpr_input": "models_train_test/jungle_rat/tprfpr_jungle_rat.csv",

    "output": "results/jungle_rat_%s.csv",
    "out_model": "models_train_test/jungle_rat/model_jungle_rat.pkl",
    "out_train_dt": "models_train_test/jungle_rat/train_data_jungle_rat.csv",
    "out_test_dt": "models_train_test/jungle_rat/test_data_jungle_rat.csv",
    "out_scores": "models_train_test/jungle_rat/scores_training_jungle_rat.csv",
    "output_tprfpr": "models_train_test/jungle_rat/tprfpr_jungle_rat.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "kin8nm": {
    "input": "datasets/kin8nm.csv",
    "model_input": "models_train_test/kin8nm/model_kin8nm.pkl",
    "train_dt_input": "models_train_test/kin8nm/train_data_kin8nm.csv",
    "test_dt_input": "models_train_test/kin8nm/test_data_kin8nm.csv",
    "scores_input": "models_train_test/kin8nm/scores_training_kin8nm.csv",
    "tprfpr_input": "models_train_test/kin8nm/tprfpr_kin8nm.csv",

    "output": "results/kin8nm_%s.csv",
    "out_model": "models_train_test/kin8nm/model_kin8nm.pkl",
    "out_train_dt": "models_train_test/kin8nm/train_data_kin8nm.csv",
    "out_test_dt": "models_train_test/kin8nm/test_data_kin8nm.csv",
    "out_scores": "models_train_test/kin8nm/scores_training_kin8nm.csv",
    "output_tprfpr": "models_train_test/kin8nm/tprfpr_kin8nm.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "kr_vs_kp": {
    "input": "datasets/kr_vs_kp.csv",
    "model_input": "models_train_test/kr_vs_kp/model_kr_vs_kp.pkl",
    "train_dt_input": "models_train_test/kr_vs_kp/train_data_kr_vs_kp.csv",
    "test_dt_input": "models_train_test/kr_vs_kp/test_data_kr_vs_kp.csv",
    "scores_input": "models_train_test/kr_vs_kp/scores_training_kr_vs_kp.csv",
    "tprfpr_input": "models_train_test/kr_vs_kp/tprfpr_kr_vs_kp.csv",

    "output": "results/kr_vs_kp_%s.csv",
    "out_model": "models_train_test/kr_vs_kp/model_kr_vs_kp.pkl",
    "out_train_dt": "models_train_test/kr_vs_kp/train_data_kr_vs_kp.csv",
    "out_test_dt": "models_train_test/kr_vs_kp/test_data_kr_vs_kp.csv",
    "out_scores": "models_train_test/kr_vs_kp/scores_training_kr_vs_kp.csv",
    "output_tprfpr": "models_train_test/kr_vs_kp/tprfpr_kr_vs_kp.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "madeline": {
    "input": "datasets/madeline.csv",
    "model_input": "models_train_test/madeline/model_madeline.pkl",
    "train_dt_input": "models_train_test/madeline/train_data_madeline.csv",
    "test_dt_input": "models_train_test/madeline/test_data_madeline.csv",
    "scores_input": "models_train_test/madeline/scores_training_madeline.csv",
    "tprfpr_input": "models_train_test/madeline/tprfpr_madeline.csv",

    "output": "results/madeline_%s.csv",
    "out_model": "models_train_test/madeline/model_madeline.pkl",
    "out_train_dt": "models_train_test/madeline/train_data_madeline.csv",
    "out_test_dt": "models_train_test/madeline/test_data_madeline.csv",
    "out_scores": "models_train_test/madeline/scores_training_madeline.csv",
    "output_tprfpr": "models_train_test/madeline/tprfpr_madeline.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "MagicTelescope": {
    "input": "datasets/MagicTelescope.csv",
    "model_input": "models_train_test/MagicTelescope/model_MagicTelescope.pkl",
    "train_dt_input": "models_train_test/MagicTelescope/train_data_MagicTelescope.csv",
    "test_dt_input": "models_train_test/MagicTelescope/test_data_MagicTelescope.csv",
    "scores_input": "models_train_test/MagicTelescope/scores_training_MagicTelescope.csv",
    "tprfpr_input": "models_train_test/MagicTelescope/tprfpr_MagicTelescope.csv",

    "output": "results/MagicTelescope_%s.csv",
    "out_model": "models_train_test/MagicTelescope/model_MagicTelescope.pkl",
    "out_train_dt": "models_train_test/MagicTelescope/train_data_MagicTelescope.csv",
    "out_test_dt": "models_train_test/MagicTelescope/test_data_MagicTelescope.csv",
    "out_scores": "models_train_test/MagicTelescope/scores_training_MagicTelescope.csv",
    "output_tprfpr": "models_train_test/MagicTelescope/tprfpr_MagicTelescope.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "mv": {
    "input": "datasets/mv.csv",
    "model_input": "models_train_test/mv/model_mv.pkl",
    "train_dt_input": "models_train_test/mv/train_data_mv.csv",
    "test_dt_input": "models_train_test/mv/test_data_mv.csv",
    "scores_input": "models_train_test/mv/scores_training_mv.csv",
    "tprfpr_input": "models_train_test/mv/tprfpr_mv.csv",

    "output": "results/mv_%s.csv",
    "out_model": "models_train_test/mv/model_mv.pkl",
    "out_train_dt": "models_train_test/mv/train_data_mv.csv",
    "out_test_dt": "models_train_test/mv/test_data_mv.csv",
    "out_scores": "models_train_test/mv/scores_training_mv.csv",
    "output_tprfpr": "models_train_test/mv/tprfpr_mv.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "numerai28_6": {
    "input": "datasets/numerai28_6.csv",
    "model_input": "models_train_test/numerai28_6/model_numerai28_6.pkl",
    "train_dt_input": "models_train_test/numerai28_6/train_data_numerai28_6.csv",
    "test_dt_input": "models_train_test/numerai28_6/test_data_numerai28_6.csv",
    "scores_input": "models_train_test/numerai28_6/scores_training_numerai28_6.csv",
    "tprfpr_input": "models_train_test/numerai28_6/tprfpr_numerai28_6.csv",

    "output": "results/numerai28_6_%s.csv",
    "out_model": "models_train_test/numerai28_6/model_numerai28_6.pkl",
    "out_train_dt": "models_train_test/numerai28_6/train_data_numerai28_6.csv",
    "out_test_dt": "models_train_test/numerai28_6/test_data_numerai28_6.csv",
    "out_scores": "models_train_test/numerai28_6/scores_training_numerai28_6.csv",
    "output_tprfpr": "models_train_test/numerai28_6/tprfpr_numerai28_6.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "nursery": {
    "input": "datasets/nursery.csv",
    "model_input": "models_train_test/nursery/model_nursery.pkl",
    "train_dt_input": "models_train_test/nursery/train_data_nursery.csv",
    "test_dt_input": "models_train_test/nursery/test_data_nursery.csv",
    "scores_input": "models_train_test/nursery/scores_training_nursery.csv",
    "tprfpr_input": "models_train_test/nursery/tprfpr_nursery.csv",

    "output": "results/nursery_%s.csv",
    "out_model": "models_train_test/nursery/model_nursery.pkl",
    "out_train_dt": "models_train_test/nursery/train_data_nursery.csv",
    "out_test_dt": "models_train_test/nursery/test_data_nursery.csv",
    "out_scores": "models_train_test/nursery/scores_training_nursery.csv",
    "output_tprfpr": "models_train_test/nursery/tprfpr_nursery.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "optdigits": {
    "input": "datasets/optdigits.csv",
    "model_input": "models_train_test/optdigits/model_optdigits.pkl",
    "train_dt_input": "models_train_test/optdigits/train_data_optdigits.csv",
    "test_dt_input": "models_train_test/optdigits/test_data_optdigits.csv",
    "scores_input": "models_train_test/optdigits/scores_training_optdigits.csv",
    "tprfpr_input": "models_train_test/optdigits/tprfpr_optdigits.csv",

    "output": "results/optdigits_%s.csv",
    "out_model": "models_train_test/optdigits/model_optdigits.pkl",
    "out_train_dt": "models_train_test/optdigits/train_data_optdigits.csv",
    "out_test_dt": "models_train_test/optdigits/test_data_optdigits.csv",
    "out_scores": "models_train_test/optdigits/scores_training_optdigits.csv",
    "output_tprfpr": "models_train_test/optdigits/tprfpr_optdigits.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "page_blocks": {
    "input": "datasets/page_blocks.csv",
    "model_input": "models_train_test/page_blocks/model_page_blocks.pkl",
    "train_dt_input": "models_train_test/page_blocks/train_data_page_blocks.csv",
    "test_dt_input": "models_train_test/page_blocks/test_data_page_blocks.csv",
    "scores_input": "models_train_test/page_blocks/scores_training_page_blocks.csv",
    "tprfpr_input": "models_train_test/page_blocks/tprfpr_page_blocks.csv",

    "output": "results/page_blocks_%s.csv",
    "out_model": "models_train_test/page_blocks/model_page_blocks.pkl",
    "out_train_dt": "models_train_test/page_blocks/train_data_page_blocks.csv",
    "out_test_dt": "models_train_test/page_blocks/test_data_page_blocks.csv",
    "out_scores": "models_train_test/page_blocks/scores_training_page_blocks.csv",
    "output_tprfpr": "models_train_test/page_blocks/tprfpr_page_blocks.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "parity5_plus_5": {
    "input": "datasets/parity5_plus_5.csv",
    "model_input": "models_train_test/parity5_plus_5/model_parity5_plus_5.pkl",
    "train_dt_input": "models_train_test/parity5_plus_5/train_data_parity5_plus_5.csv",
    "test_dt_input": "models_train_test/parity5_plus_5/test_data_parity5_plus_5.csv",
    "scores_input": "models_train_test/parity5_plus_5/scores_training_parity5_plus_5.csv",
    "tprfpr_input": "models_train_test/parity5_plus_5/tprfpr_parity5_plus_5.csv",

    "output": "results/parity5_plus_5_%s.csv",
    "out_model": "models_train_test/parity5_plus_5/model_parity5_plus_5.pkl",
    "out_train_dt": "models_train_test/parity5_plus_5/train_data_parity5_plus_5.csv",
    "out_test_dt": "models_train_test/parity5_plus_5/test_data_parity5_plus_5.csv",
    "out_scores": "models_train_test/parity5_plus_5/scores_training_parity5_plus_5.csv",
    "output_tprfpr": "models_train_test/parity5_plus_5/tprfpr_parity5_plus_5.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "pendigits": {
    "input": "datasets/pendigits.csv",
    "model_input": "models_train_test/pendigits/model_pendigits.pkl",
    "train_dt_input": "models_train_test/pendigits/train_data_pendigits.csv",
    "test_dt_input": "models_train_test/pendigits/test_data_pendigits.csv",
    "scores_input": "models_train_test/pendigits/scores_training_pendigits.csv",
    "tprfpr_input": "models_train_test/pendigits/tprfpr_pendigits.csv",

    "output": "results/pendigits_%s.csv",
    "out_model": "models_train_test/pendigits/model_pendigits.pkl",
    "out_train_dt": "models_train_test/pendigits/train_data_pendigits.csv",
    "out_test_dt": "models_train_test/pendigits/test_data_pendigits.csv",
    "out_scores": "models_train_test/pendigits/scores_training_pendigits.csv",
    "output_tprfpr": "models_train_test/pendigits/tprfpr_pendigits.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "philippine": {
    "input": "datasets/philippine.csv",
    "model_input": "models_train_test/philippine/model_philippine.pkl",
    "train_dt_input": "models_train_test/philippine/train_data_philippine.csv",
    "test_dt_input": "models_train_test/philippine/test_data_philippine.csv",
    "scores_input": "models_train_test/philippine/scores_training_philippine.csv",
    "tprfpr_input": "models_train_test/philippine/tprfpr_philippine.csv",

    "output": "results/philippine_%s.csv",
    "out_model": "models_train_test/philippine/model_philippine.pkl",
    "out_train_dt": "models_train_test/philippine/train_data_philippine.csv",
    "out_test_dt": "models_train_test/philippine/test_data_philippine.csv",
    "out_scores": "models_train_test/philippine/scores_training_philippine.csv",
    "output_tprfpr": "models_train_test/philippine/tprfpr_philippine.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "PhishingWebsites": {
    "input": "datasets/PhishingWebsites.csv",
    "model_input": "models_train_test/PhishingWebsites/model_PhishingWebsites.pkl",
    "train_dt_input": "models_train_test/PhishingWebsites/train_data_PhishingWebsites.csv",
    "test_dt_input": "models_train_test/PhishingWebsites/test_data_PhishingWebsites.csv",
    "scores_input": "models_train_test/PhishingWebsites/scores_training_PhishingWebsites.csv",
    "tprfpr_input": "models_train_test/PhishingWebsites/tprfpr_PhishingWebsites.csv",

    "output": "results/PhishingWebsites_%s.csv",
    "out_model": "models_train_test/PhishingWebsites/model_PhishingWebsites.pkl",
    "out_train_dt": "models_train_test/PhishingWebsites/train_data_PhishingWebsites.csv",
    "out_test_dt": "models_train_test/PhishingWebsites/test_data_PhishingWebsites.csv",
    "out_scores": "models_train_test/PhishingWebsites/scores_training_PhishingWebsites.csv",
    "output_tprfpr": "models_train_test/PhishingWebsites/tprfpr_PhishingWebsites.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "pol": {
    "input": "datasets/pol.csv",
    "model_input": "models_train_test/pol/model_pol.pkl",
    "train_dt_input": "models_train_test/pol/train_data_pol.csv",
    "test_dt_input": "models_train_test/pol/test_data_pol.csv",
    "scores_input": "models_train_test/pol/scores_training_pol.csv",
    "tprfpr_input": "models_train_test/pol/tprfpr_pol.csv",

    "output": "results/pol_%s.csv",
    "out_model": "models_train_test/pol/model_pol.pkl",
    "out_train_dt": "models_train_test/pol/train_data_pol.csv",
    "out_test_dt": "models_train_test/pol/test_data_pol.csv",
    "out_scores": "models_train_test/pol/scores_training_pol.csv",
    "output_tprfpr": "models_train_test/pol/tprfpr_pol.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "puma8NH": {
    "input": "datasets/puma8NH.csv",
    "model_input": "models_train_test/puma8NH/model_puma8NH.pkl",
    "train_dt_input": "models_train_test/puma8NH/train_data_puma8NH.csv",
    "test_dt_input": "models_train_test/puma8NH/test_data_puma8NH.csv",
    "scores_input": "models_train_test/puma8NH/scores_training_puma8NH.csv",
    "tprfpr_input": "models_train_test/puma8NH/tprfpr_puma8NH.csv",

    "output": "results/puma8NH_%s.csv",
    "out_model": "models_train_test/puma8NH/model_puma8NH.pkl",
    "out_train_dt": "models_train_test/puma8NH/train_data_puma8NH.csv",
    "out_test_dt": "models_train_test/puma8NH/test_data_puma8NH.csv",
    "out_scores": "models_train_test/puma8NH/scores_training_puma8NH.csv",
    "output_tprfpr": "models_train_test/puma8NH/tprfpr_puma8NH.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "puma32H": {
    "input": "datasets/puma32H.csv",
    "model_input": "models_train_test/puma32H/model_puma32H.pkl",
    "train_dt_input": "models_train_test/puma32H/train_data_puma32H.csv",
    "test_dt_input": "models_train_test/puma32H/test_data_puma32H.csv",
    "scores_input": "models_train_test/puma32H/scores_training_puma32H.csv",
    "tprfpr_input": "models_train_test/puma32H/tprfpr_puma32H.csv",

    "output": "results/puma32H_%s.csv",
    "out_model": "models_train_test/puma32H/model_puma32H.pkl",
    "out_train_dt": "models_train_test/puma32H/train_data_puma32H.csv",
    "out_test_dt": "models_train_test/puma32H/test_data_puma32H.csv",
    "out_scores": "models_train_test/puma32H/scores_training_puma32H.csv",
    "output_tprfpr": "models_train_test/puma32H/tprfpr_puma32H.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "quake": {
    "input": "datasets/quake.csv",
    "model_input": "models_train_test/quake/model_quake.pkl",
    "train_dt_input": "models_train_test/quake/train_data_quake.csv",
    "test_dt_input": "models_train_test/quake/test_data_quake.csv",
    "scores_input": "models_train_test/quake/scores_training_quake.csv",
    "tprfpr_input": "models_train_test/quake/tprfpr_quake.csv",

    "output": "results/quake_%s.csv",
    "out_model": "models_train_test/quake/model_quake.pkl",
    "out_train_dt": "models_train_test/quake/train_data_quake.csv",
    "out_test_dt": "models_train_test/quake/test_data_quake.csv",
    "out_scores": "models_train_test/quake/scores_training_quake.csv",
    "output_tprfpr": "models_train_test/quake/tprfpr_quake.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "ringnorm": {
    "input": "datasets/ringnorm.csv",
    "model_input": "models_train_test/ringnorm/model_ringnorm.pkl",
    "train_dt_input": "models_train_test/ringnorm/train_data_ringnorm.csv",
    "test_dt_input": "models_train_test/ringnorm/test_data_ringnorm.csv",
    "scores_input": "models_train_test/ringnorm/scores_training_ringnorm.csv",
    "tprfpr_input": "models_train_test/ringnorm/tprfpr_ringnorm.csv",

    "output": "results/ringnorm_%s.csv",
    "out_model": "models_train_test/ringnorm/model_ringnorm.pkl",
    "out_train_dt": "models_train_test/ringnorm/train_data_ringnorm.csv",
    "out_test_dt": "models_train_test/ringnorm/test_data_ringnorm.csv",
    "out_scores": "models_train_test/ringnorm/scores_training_ringnorm.csv",
    "output_tprfpr": "models_train_test/ringnorm/tprfpr_ringnorm.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "rmftsa_sleepdata": {
    "input": "datasets/rmftsa_sleepdata.csv",
    "model_input": "models_train_test/rmftsa_sleepdata/model_rmftsa_sleepdata.pkl",
    "train_dt_input": "models_train_test/rmftsa_sleepdata/train_data_rmftsa_sleepdata.csv",
    "test_dt_input": "models_train_test/rmftsa_sleepdata/test_data_rmftsa_sleepdata.csv",
    "scores_input": "models_train_test/rmftsa_sleepdata/scores_training_rmftsa_sleepdata.csv",
    "tprfpr_input": "models_train_test/rmftsa_sleepdata/tprfpr_rmftsa_sleepdata.csv",

    "output": "results/rmftsa_sleepdata_%s.csv",
    "out_model": "models_train_test/rmftsa_sleepdata/model_rmftsa_sleepdata.pkl",
    "out_train_dt": "models_train_test/rmftsa_sleepdata/train_data_rmftsa_sleepdata.csv",
    "out_test_dt": "models_train_test/rmftsa_sleepdata/test_data_rmftsa_sleepdata.csv",
    "out_scores": "models_train_test/rmftsa_sleepdata/scores_training_rmftsa_sleepdata.csv",
    "output_tprfpr": "models_train_test/rmftsa_sleepdata/tprfpr_rmftsa_sleepdata.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "Run_or_walk_information": {
    "input": "datasets/Run_or_walk_information.csv",
    "model_input": "models_train_test/Run_or_walk_information/model_Run_or_walk_information.pkl",
    "train_dt_input": "models_train_test/Run_or_walk_information/train_data_Run_or_walk_information.csv",
    "test_dt_input": "models_train_test/Run_or_walk_information/test_data_Run_or_walk_information.csv",
    "scores_input": "models_train_test/Run_or_walk_information/scores_training_Run_or_walk_information.csv",
    "tprfpr_input": "models_train_test/Run_or_walk_information/tprfpr_Run_or_walk_information.csv",

    "output": "results/Run_or_walk_information_%s.csv",
    "out_model": "models_train_test/Run_or_walk_information/model_Run_or_walk_information.pkl",
    "out_train_dt": "models_train_test/Run_or_walk_information/train_data_Run_or_walk_information.csv",
    "out_test_dt": "models_train_test/Run_or_walk_information/test_data_Run_or_walk_information.csv",
    "out_scores": "models_train_test/Run_or_walk_information/scores_training_Run_or_walk_information.csv",
    "output_tprfpr": "models_train_test/Run_or_walk_information/tprfpr_Run_or_walk_information.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "scene": {
    "input": "datasets/scene.csv",
    "model_input": "models_train_test/scene/model_scene.pkl",
    "train_dt_input": "models_train_test/scene/train_data_scene.csv",
    "test_dt_input": "models_train_test/scene/test_data_scene.csv",
    "scores_input": "models_train_test/scene/scores_training_scene.csv",
    "tprfpr_input": "models_train_test/scene/tprfpr_scene.csv",

    "output": "results/scene_%s.csv",
    "out_model": "models_train_test/scene/model_scene.pkl",
    "out_train_dt": "models_train_test/scene/train_data_scene.csv",
    "out_test_dt": "models_train_test/scene/test_data_scene.csv",
    "out_scores": "models_train_test/scene/scores_training_scene.csv",
    "output_tprfpr": "models_train_test/scene/tprfpr_scene.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "space_ga": {
    "input": "datasets/space_ga.csv",
    "model_input": "models_train_test/space_ga/model_space_ga.pkl",
    "train_dt_input": "models_train_test/space_ga/train_data_space_ga.csv",
    "test_dt_input": "models_train_test/space_ga/test_data_space_ga.csv",
    "scores_input": "models_train_test/space_ga/scores_training_space_ga.csv",
    "tprfpr_input": "models_train_test/space_ga/tprfpr_space_ga.csv",

    "output": "results/space_ga_%s.csv",
    "out_model": "models_train_test/space_ga/model_space_ga.pkl",
    "out_train_dt": "models_train_test/space_ga/train_data_space_ga.csv",
    "out_test_dt": "models_train_test/space_ga/test_data_space_ga.csv",
    "out_scores": "models_train_test/space_ga/scores_training_space_ga.csv",
    "output_tprfpr": "models_train_test/space_ga/tprfpr_space_ga.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "splice": {
    "input": "datasets/splice.csv",
    "model_input": "models_train_test/splice/model_splice.pkl",
    "train_dt_input": "models_train_test/splice/train_data_splice.csv",
    "test_dt_input": "models_train_test/splice/test_data_splice.csv",
    "scores_input": "models_train_test/splice/scores_training_splice.csv",
    "tprfpr_input": "models_train_test/splice/tprfpr_splice.csv",

    "output": "results/splice_%s.csv",
    "out_model": "models_train_test/splice/model_splice.pkl",
    "out_train_dt": "models_train_test/splice/train_data_splice.csv",
    "out_test_dt": "models_train_test/splice/test_data_splice.csv",
    "out_scores": "models_train_test/splice/scores_training_splice.csv",
    "output_tprfpr": "models_train_test/splice/tprfpr_splice.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "steel_plates_fault": {
    "input": "datasets/steel_plates_fault.csv",
    "model_input": "models_train_test/steel_plates_fault/model_steel_plates_fault.pkl",
    "train_dt_input": "models_train_test/steel_plates_fault/train_data_steel_plates_fault.csv",
    "test_dt_input": "models_train_test/steel_plates_fault/test_data_steel_plates_fault.csv",
    "scores_input": "models_train_test/steel_plates_fault/scores_training_steel_plates_fault.csv",
    "tprfpr_input": "models_train_test/steel_plates_fault/tprfpr_steel_plates_fault.csv",

    "output": "results/steel_plates_fault_%s.csv",
    "out_model": "models_train_test/steel_plates_fault/model_steel_plates_fault.pkl",
    "out_train_dt": "models_train_test/steel_plates_fault/train_data_steel_plates_fault.csv",
    "out_test_dt": "models_train_test/steel_plates_fault/test_data_steel_plates_fault.csv",
    "out_scores": "models_train_test/steel_plates_fault/scores_training_steel_plates_fault.csv",
    "output_tprfpr": "models_train_test/steel_plates_fault/tprfpr_steel_plates_fault.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "stock_arff": {
    "input": "datasets/stock_arff.csv",
    "model_input": "models_train_test/stock_arff/model_stock_arff.pkl",
    "train_dt_input": "models_train_test/stock_arff/train_data_stock_arff.csv",
    "test_dt_input": "models_train_test/stock_arff/test_data_stock_arff.csv",
    "scores_input": "models_train_test/stock_arff/scores_training_stock_arff.csv",
    "tprfpr_input": "models_train_test/stock_arff/tprfpr_stock_arff.csv",

    "output": "results/stock_arff_%s.csv",
    "out_model": "models_train_test/stock_arff/model_stock_arff.pkl",
    "out_train_dt": "models_train_test/stock_arff/train_data_stock_arff.csv",
    "out_test_dt": "models_train_test/stock_arff/test_data_stock_arff.csv",
    "out_scores": "models_train_test/stock_arff/scores_training_stock_arff.csv",
    "output_tprfpr": "models_train_test/stock_arff/tprfpr_stock_arff.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "sylva_prior": {
    "input": "datasets/sylva_prior.csv",
    "model_input": "models_train_test/sylva_prior/model_sylva_prior.pkl",
    "train_dt_input": "models_train_test/sylva_prior/train_data_sylva_prior.csv",
    "test_dt_input": "models_train_test/sylva_prior/test_data_sylva_prior.csv",
    "scores_input": "models_train_test/sylva_prior/scores_training_sylva_prior.csv",
    "tprfpr_input": "models_train_test/sylva_prior/tprfpr_sylva_prior.csv",

    "output": "results/sylva_prior_%s.csv",
    "out_model": "models_train_test/sylva_prior/model_sylva_prior.pkl",
    "out_train_dt": "models_train_test/sylva_prior/train_data_sylva_prior.csv",
    "out_test_dt": "models_train_test/sylva_prior/test_data_sylva_prior.csv",
    "out_scores": "models_train_test/sylva_prior/scores_training_sylva_prior.csv",
    "output_tprfpr": "models_train_test/sylva_prior/tprfpr_sylva_prior.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "sylvine": {
    "input": "datasets/sylvine.csv",
    "model_input": "models_train_test/sylvine/model_sylvine.pkl",
    "train_dt_input": "models_train_test/sylvine/train_data_sylvine.csv",
    "test_dt_input": "models_train_test/sylvine/test_data_sylvine.csv",
    "scores_input": "models_train_test/sylvine/scores_training_sylvine.csv",
    "tprfpr_input": "models_train_test/sylvine/tprfpr_sylvine.csv",

    "output": "results/sylvine_%s.csv",
    "out_model": "models_train_test/sylvine/model_sylvine.pkl",
    "out_train_dt": "models_train_test/sylvine/train_data_sylvine.csv",
    "out_test_dt": "models_train_test/sylvine/test_data_sylvine.csv",
    "out_scores": "models_train_test/sylvine/scores_training_sylvine.csv",
    "output_tprfpr": "models_train_test/sylvine/tprfpr_sylvine.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "test_dataset": {
    "input": "datasets/test_dataset.csv",
    "model_input": "models_train_test/test_dataset/model_test_dataset.pkl",
    "train_dt_input": "models_train_test/test_dataset/train_data_test_dataset.csv",
    "test_dt_input": "models_train_test/test_dataset/test_data_test_dataset.csv",
    "scores_input": "models_train_test/test_dataset/scores_training_test_dataset.csv",
    "tprfpr_input": "models_train_test/test_dataset/tprfpr_test_dataset.csv",

    "output": "results/test_dataset_%s.csv",
    "out_model": "models_train_test/test_dataset/model_test_dataset.pkl",
    "out_train_dt": "models_train_test/test_dataset/train_data_test_dataset.csv",
    "out_test_dt": "models_train_test/test_dataset/test_data_test_dataset.csv",
    "out_scores": "models_train_test/test_dataset/scores_training_test_dataset.csv",
    "output_tprfpr": "models_train_test/test_dataset/tprfpr_test_dataset.csv",

    "class_feature": "class",
    "positive_label": 1,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "Titanic": {
    "input": "datasets/Titanic.csv",
    "model_input": "models_train_test/Titanic/model_Titanic.pkl",
    "train_dt_input": "models_train_test/Titanic/train_data_Titanic.csv",
    "test_dt_input": "models_train_test/Titanic/test_data_Titanic.csv",
    "scores_input": "models_train_test/Titanic/scores_training_Titanic.csv",
    "tprfpr_input": "models_train_test/Titanic/tprfpr_Titanic.csv",

    "output": "results/Titanic_%s.csv",
    "out_model": "models_train_test/Titanic/model_Titanic.pkl",
    "out_train_dt": "models_train_test/Titanic/train_data_Titanic.csv",
    "out_test_dt": "models_train_test/Titanic/test_data_Titanic.csv",
    "out_scores": "models_train_test/Titanic/scores_training_Titanic.csv",
    "output_tprfpr": "models_train_test/Titanic/tprfpr_Titanic.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "twonorm": {
    "input": "datasets/twonorm.csv",
    "model_input": "models_train_test/twonorm/model_twonorm.pkl",
    "train_dt_input": "models_train_test/twonorm/train_data_twonorm.csv",
    "test_dt_input": "models_train_test/twonorm/test_data_twonorm.csv",
    "scores_input": "models_train_test/twonorm/scores_training_twonorm.csv",
    "tprfpr_input": "models_train_test/twonorm/tprfpr_twonorm.csv",

    "output": "results/twonorm_%s.csv",
    "out_model": "models_train_test/twonorm/model_twonorm.pkl",
    "out_train_dt": "models_train_test/twonorm/train_data_twonorm.csv",
    "out_test_dt": "models_train_test/twonorm/test_data_twonorm.csv",
    "out_scores": "models_train_test/twonorm/scores_training_twonorm.csv",
    "output_tprfpr": "models_train_test/twonorm/tprfpr_twonorm.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "USPS": {
    "input": "datasets/USPS.csv",
    "model_input": "models_train_test/USPS/model_USPS.pkl",
    "train_dt_input": "models_train_test/USPS/train_data_USPS.csv",
    "test_dt_input": "models_train_test/USPS/test_data_USPS.csv",
    "scores_input": "models_train_test/USPS/scores_training_USPS.csv",
    "tprfpr_input": "models_train_test/USPS/tprfpr_USPS.csv",

    "output": "results/USPS_%s.csv",
    "out_model": "models_train_test/USPS/model_USPS.pkl",
    "out_train_dt": "models_train_test/USPS/train_data_USPS.csv",
    "out_test_dt": "models_train_test/USPS/test_data_USPS.csv",
    "out_scores": "models_train_test/USPS/scores_training_USPS.csv",
    "output_tprfpr": "models_train_test/USPS/tprfpr_USPS.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },
  "visualizing_soil": {
    "input": "datasets/visualizing_soil.csv",
    "model_input": "models_train_test/visualizing_soil/model_visualizing_soil.pkl",
    "train_dt_input": "models_train_test/visualizing_soil/train_data_visualizing_soil.csv",
    "test_dt_input": "models_train_test/visualizing_soil/test_data_visualizing_soil.csv",
    "scores_input": "models_train_test/visualizing_soil/scores_training_visualizing_soil.csv",
    "tprfpr_input": "models_train_test/visualizing_soil/tprfpr_visualizing_soil.csv",

    "output": "results/visualizing_soil_%s.csv",
    "out_model": "models_train_test/visualizing_soil/model_visualizing_soil.pkl",
    "out_train_dt": "models_train_test/visualizing_soil/train_data_visualizing_soil.csv",
    "out_test_dt": "models_train_test/visualizing_soil/test_data_visualizing_soil.csv",
    "out_scores": "models_train_test/visualizing_soil/scores_training_visualizing_soil.csv",
    "output_tprfpr": "models_train_test/visualizing_soil/tprfpr_visualizing_soil.csv",

    "class_feature": "class",
    "positive_label": 2,
    "negative_labels": None, # all other labels
    "features": None,
  },

}