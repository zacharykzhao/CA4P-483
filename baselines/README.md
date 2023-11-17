This folder provides the reproducibility process of baseline methods used on our dataset:

1. For CRF, HMM, Bi-LSTM, Bi-LSTM-CRF:

   1. please make sure the requirements in './named_entity_recoginition-master/requirement.txt' have been satisfied.

   2. ```
      python ./named_entity_recoginition-master/CSP3_main.py 
      ```

2. For BERT-BiLSTM-CRF:

   1. please make sure the requirements in './BERT-BiLSTM-CRF-NER-tf-master/requirement.txt' have been satisfied and following the './BERT-BiLSTM-CRF-NER-tf-master/README.md' to download the pre-trained model, modify the bert path in 'sh ./named_entity_recognition-master/run.sh', then:

   2. ```
      sh ./named_entity_recognition-master/run.sh
      ```

3. For LatticeLSTM:

   1. Please make sure the requirements in './Batch_Parallel_LatticeLSTM-master/requirement.txt', follow the '/Batch_Parallel_LatticeLSTM-master/README.md' to download the embeddings, modify the paths in './Batch_Parallel_LatticeLSTM-master/pathes.py', then

      ```
      python ./Batch_Parallel_LatticeLSTM-master/main_CSP3.py
      ```

      