# CA4P-483

Public Repository: https://github.com/zacharykzhao/CA4P-483



#### 1. Download the repository

Run the following script in your terminal (please make sure ***git***(https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) has been correctly installed):

```shell
git clone https://github.com/zacharykzhao/CA4P-483
```

<img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 12.22.31.png" alt="截屏2023-11-16 12.22.31" style="zoom:50%;" />

<img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 12.22.50.png" alt="截屏2023-11-16 12.22.50" style="zoom:50%;" />

#### 2. Setup

Execute the following script in command line to create the python interpreter environment to reproduce the results:

```
conda create -n ca4p python=3.6
conda activate ca4p
pip install -r ./CA4P-483/requirements.txt 

```

#### 3. Get information in Table 1

+ 3.1 Run the following script in command line to reproduce statistic information in Table 1:

```
python ./CA4P-483/DataSetStatistics/Table_1.py
```

<img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 15.39.54.png" alt="截屏2023-11-16 15.39.54" style="zoom:50%;" />

* Please notice that the "handler" in the script corresponds to "Controller" in the paper, and the "subjects" in the script conrresponds to "Receiver" in the paper.

+ 3.2 To obatian the **Kappa** agreements for each components in Table 1, please refer to the form: "./CA4P-483/Manual Kappa Evaluation 0611.xlsx".

​	In the form, row 1-9 denotes the manual check results of the 1st annotator, row 12-20 refers to the manual check results of the 2nd annotator, and row 25-33 gives the **Kappa** aggreements based on the manual check results, i.e., row 1-20.



#### 4. Reproduce results in Table 3 & 4

+ 4.1 For CRF, HMM, Bi-LSTM, Bi-LSTM-CRF:

  + 4.1.1 Execute the following script in command line to create the python interpreter  virtual environment to reproduce the results.  Besides, please make sure that the pytorch (GPU version is strongly recommended) is correctly installed in your environment.

    + ```shell
      conda create -n CHBB python=3.6
      conda activate CHBB
      pip install -r ./CA4P-483/baselines/named_entity_recognition-master/requirement.txt 
      ```

    + <img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 16.03.42.png" alt="截屏2023-11-16 16.03.42" style="zoom:50%;" />

  + 4.1.2 Execute the following script in command line to reproduce the results:

    + ```
      cd CA4P-483/baselines/named_entity_recognition-master 
      python CSP3_main.py
      ```

      + CRF Results:
        + <img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 17.56.42.png" alt="截屏2023-11-16 17.56.42" style="zoom:50%;" />
      + HMM Results:
        + <img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 17.56.57.png" alt="截屏2023-11-16 17.56.57" style="zoom:50%;" />
      + BiLSTM results:
        + <img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 18.25.59.png" alt="截屏2023-11-16 18.25.59" style="zoom:50%;" />
        + <img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 21.46.59.png" alt="截屏2023-11-16 21.46.59" style="zoom:50%;" />
        + BiLSTM-CRF
          + <img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 21.47.43.png" alt="截屏2023-11-16 21.47.43" style="zoom:50%;" />
          + <img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 21.48.01.png" alt="截屏2023-11-16 21.48.01" style="zoom:50%;" />
          + Confusion Matrix for BiLSTM-CRF (Corresponds to Figure2)

  ![截屏2023-11-16 21.48.28](/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 21.48.28.png)

  + **It is wothy noticing that HMM, CRF, BiLSTM and BiLSTM-CRF calculate the metric for each class/components with B, I, E seperately. We average them to obtain the results in Table 3.**

+ 4.2 For BERT-BiLSTM-CRF:

  + 4.2.1 Execute the following script in command line to create the python interpreter  virtual environment to reproduce the results:

    + ```
      conda create -n BBC python=3.9
      conda activate BBC
      cd ../BERT-BiLSTM-CRF-NER-tf-master 
      pip install -r requirement.txt 
      python3 setup.py install
      ```

    + <img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 18.12.03.png" alt="截屏2023-11-16 18.01.23" style="zoom:50%;" />
    + 

    + 4.2.2 Following  the './BERT-BiLSTM-CRF-NER-tf-master/README.md' to download the pre-trained model and setup BERT-BiLSTM-CRF-NER.  Besides, please make sure that the tensorflow (GPU version is strongly recommended) is correctly installed in your environment.

      + In the evaluation phase, we use the model (L=12, H=768, link: https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip ) for evaluation. 

    + 4.2.3 Please replace variable of path to pre-trained model in "run.sh" (line 5-7) to the path you downloaded the model in Step 4.2.2. 

      + ![截屏2023-11-16 18.07.31](/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 18.07.31.png)

    + 4.2.4 Execute the following script in command line to reproduce the results:

      + ```
        sh run.sh
        ```

      + <img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 20.25.05.png" alt="截屏2023-11-16 20.25.05" style="zoom:50%;" />

      + <img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-17 12.42.50.png" alt="截屏2023-11-17 12.42.50" style="zoom:50%;" />

  + 4.3 For LatticeLSTM:

    + 4.3.1 Please make sure the requirements in './Batch_Parallel_LatticeLSTM-master/requirement.txt', follow the '/Batch_Parallel_LatticeLSTM-master/README.md' to download the embeddings. Besides, please make sure that the pytorch (GPU version is strongly recommended) is correctly installed in your environment. Modify the paths in './Batch_Parallel_LatticeLSTM-master/pathes.py'.

      + ```
        cd ../
        cd Batch_Parallel_LatticeLSTM-master/
        conda create -n LaticeLSTM python=3.7.3
        conda activate LaticeLSTM
        pip install fastNLP
        pip install pytorch>=1.1.0
        pip install numpy>=1.16.4
        pip install fitlog>=0.2.0
        ```

      + ```
        python main_CSP3.py 
        ```

      + <img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 21.00.41.png" alt="截屏2023-11-16 21.00.41" style="zoom:50%;" />

      + <img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-16 21.49.53.png" alt="截屏2023-11-16 21.49.53" style="zoom:50%;" />

+ 4.4 For the **Manual Agreements** given in Table3,  please refer to the form: "./CA4P-483/Manual Kappa Evaluation 0611.xlsx".
  + In the form, row 1-9 denotes the manual check results of the 1st annotator, row 12-20 refers to the manual check results of the 2nd annotator, and row 56-64 gives the **Manual Agreements**  based on the manual check results, i.e., row 1-20.
+ **\* Please note that the above algorithms may include random processes during operation, such as random initialization parameters, which may cause the results to be slightly different from those given in the paper.**

	##### 5. Reproduce Figure.4

Run the following script in command line to reproduce Figure.4:

```
python ./CA4P-483/DataSetStatistics/Figure4.py 
```

<img src="/Users/zhaokaifa_imac/Library/Application Support/typora-user-images/截屏2023-11-17 12.45.17.png" alt="截屏2023-11-17 12.45.17" style="zoom:50%;" />