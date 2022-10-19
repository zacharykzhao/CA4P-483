#export CUDA_VISIBLE_DEVICES=1,2,
bert-base-ner-train \
	-data_dir "./data/" \
	-output_dir "./results" \
	-init_checkpoint "./uncased_L-12_H-768_A-12/bert_model.ckpt" \
	-bert_config_file "./uncased_L-12_H-768_A-12/bert_config.json" \
	-vocab_file "./uncased_L-12_H-768_A-12/vocab.txt" \
	-num_train_epochs 100 \
	-do_train=true \
	-do_eval=true \
	-do_predict=true



	
