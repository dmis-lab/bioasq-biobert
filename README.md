## Pre-trained Language Model for Biomedical Question Answering <br> _BioBERT at BioASQ 7b -Phase B_
This repository provides the source code and pre-processed datasets of our participating model for the BioASQ Challenge 7b. We utilized BioBERT, a language representation model for the biomedical domain, with minimum modifications for the challenge. 
<br>Please refer to our paper [Pre-trained Language Model for Biomedical Question Answering](https://arxiv.org/abs/1909.08229) for more details.
This paper is accepted for an oral presentation in **BioASQ Workshop @ ECML PKDD 2019**.

## Citation

Please cite [the published version of the paper](https://link.springer.com/chapter/10.1007/978-3-030-43887-6_64):
```
@InProceedings{10.1007/978-3-030-43887-6_64,
  author="Yoon, Wonjin and Lee, Jinhyuk and Kim, Donghyeon and Jeong, Minbyul and Kang, Jaewoo",
  editor="Cellier, Peggy and Driessens, Kurt",
  title="Pre-trained Language Model for Biomedical Question Answering",
  booktitle="Machine Learning and Knowledge Discovery in Databases",
  year="2020",
  publisher="Springer International Publishing",
  address="Cham",
  pages="727--740",
  isbn="978-3-030-43887-6"
}
```

Also, we wish you to cite [BioBERT paper](http://dx.doi.org/10.1093/bioinformatics/btz682) as well since our model is based on BioBERT pre-trained weight. 
```
@article{lee2019biobert,
  title={BioBERT: a pre-trained biomedical language representation model for biomedical text mining},
  author={Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
  doi = {10.1093/bioinformatics/btz682}, 
  journal={Bioinformatics},
  year={2019}
}
```

## Installation
Please note that this repository is based on the [BioBERT repository](https://github.com/dmis-lab/biobert).

### Pre-trained weights (Pre-trained on SQuAD)
We are releasing the pre-trained weights for BioBERT system in the paper. The weights are pre-trained on `SQuAD v1.1` or `SQuAD v2.0` dataset on top of `BioBERT v1.1`(1M steps pre-trained on PubMed corpus).
We only used training set of SQuAD datasets. 
<br>For best performance, please use `BioBERT v1.1-SQuAD v1.1` for factoid and list questions and `BioBERT v1.1-SQuAD v2.0` for yseno questions.
*   **[`BioBERT v1.1 - SQuAD v1.1`](https://drive.google.com/open?id=1rXFQRcV69QHAxghQ3NeAlhkg6ykpflVK)** : Recommanded for factoid and list questions
<br>`SHA-1 Checksum : 408809150A23B4B99EFD21AF2B4ACEA52B31F3D9`
*   **[`BioBERT v1.1 - SQuAD v2.0`](https://drive.google.com/open?id=1AR6CLa17oMjdnYtV1xF3w9GygSrElmxK)** : Recommanded for yseno questions
<br>`SHA-1 Checksum : 9A10621691BFEB834CBFD5F81E9D2C099247803A`
*   **[`bert_config.json`](https://drive.google.com/open?id=17fX1-oChZ5rxu-e-JuaZl2I96q1dGJO4) [`vocab.txt`](https://drive.google.com/open?id=1GQUvBbXvlI_PeUPsZTqh7xQDZMOXh7ko)** : Essential files.

As an alternative option, you may wish to pre-train from scratch. In that case, please follow :
```
1. Fine-tune BioBERT on SQuAD dataset
2. Use the resulting ckpt of 1 as an initial checkpoint for fine-tuning BioASQ datasets. 
```
Be sure to set the output folder of step 2 as a different folder of step 1.

## Datasets
We provide pre-processed version of BioASQ 6b/7b - Phase B datasets for each task as follows:
*   **[`BioASQ 6b/7b`](https://drive.google.com/open?id=1-KzAQzaE-Zd4jOlZG_7k7D4odqPI3dL1)** (23 MB) Last update : 15th Oct. 2019 

Due to the copyright issue, we can not provide golden answers for BioASQ 6b test dataset at the moment. 
**However, you can extract golden answers for 6b from original BioASQ 7b dataset.**
To use original BioASQ datasets, you should register in [BioASQ website](http://participants-area.bioasq.org). 

For details on the datasets, please see **An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition (Tsatsaronis et al. 2015)**.

## Fine-tuning BioBERT
After downloading one of the pre-trained models, unpack it to any directory you want, which we will denote as `$BIOBERT_DIR`.
You need to download other essential files ([`bert_config.json`](https://drive.google.com/open?id=17fX1-oChZ5rxu-e-JuaZl2I96q1dGJO4) and [`vocab.txt`](https://drive.google.com/open?id=1GQUvBbXvlI_PeUPsZTqh7xQDZMOXh7ko)) to `$BIOBERT_DIR` as well. 

Please download our pre-processed version of BioASQ-6/7b datasets, and unpack it to `$BIOASQ_DIR`.

### Training and predicting

Please use `run_factoid.py`, `run_yesno.py` and `run_list.py` for yesno, factoid and list questions respectively.
Use `BioASQ-*.json` as training and testing dataset which we pre-processed the original BioASQ data to SQuAD dataset form. 
This is necessary as the input data format of BioBERT is different from BioASQ dataset format. 
Also, please be informed that the do_lower_case flag should be set as `--do_lower_case=False` since BioBERT model is based on `BERT-BASE (CASED)` model. 

As an example, the following command runs fine-tuning and predicting code on factoid questions (6b; _full abstract_ method) with default arguments.
<br>Please see [examplecode.sh](examplecode.sh) for yesno and list questions.

``` 
export BIOBERT_DIR=$HOME/BioASQ/BERT-pubmed-1000000-SQuAD
export BIOASQ_DIR=$HOME/BioASQ/data-release

python run_factoid.py \
     --do_train=True \
     --do_predict=True \
     --vocab_file=$BIOBERT_DIR/vocab.txt \
     --bert_config_file=$BIOBERT_DIR/bert_config.json \
     --init_checkpoint=$BIOBERT_DIR/model.ckpt-14599 \
     --max_seq_length=384 \
     --train_batch_size=12 \
     --learning_rate=5e-6 \
     --doc_stride=128 \
     --num_train_epochs=5.0 \
     --do_lower_case=False \
     --train_file=$BIOASQ_DIR/BioASQ-6b/train/Full-Abstract/BioASQ-train-factoid-6b-full-annotated.json \
     --predict_file=$BIOASQ_DIR/BioASQ-6b/test/Full-Abstract/BioASQ-test-factoid-6b-3.json \
     --output_dir=/tmp/factoid_output/
```
You can change the arguments as you want. Once you have trained your model, you can use it in inference mode by using `--do_train=false --do_predict=true` for evaluating other json file with identical structure.

The predictions will be saved into a file called `predictions.json` and `nbest_predictions.json` in the `output_dir`.
Run transform file (for example, `transform_n2b_factoid.py`) in `./biocodes/` folder to convert `nbest_predictions.json` or `predictions.json` to BioASQ JSON format, which will be used for the official evaluation.
```
python ./biocodes/transform_n2b_factoid.py --nbest_path={QA_output_dir}/nbest_predictions.json --output_path={output_dir}
python ./biocodes/transform_n2b_yesno.py --nbest_path={QA_output_dir}/predictions.json --output_path={output_dir}
python ./biocodes/transform_n2b_list.py --nbest_path={QA_output_dir}/nbest_predictions.json --output_path={output_dir}
```
This will generate `BioASQform_BioASQ-answer.json` in `{output_dir}`.
Clone **[`evaluation code`](https://github.com/BioASQ/Evaluation-Measures)** from BioASQ github and run evaluation code on `Evaluation-Measures` directory. Please note that you should put 5 as parameter for -e if you are evaluating the system for BioASQ 5b/6b/7b dataset .
```
cd Evaluation-Measures
java -Xmx10G -cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 \
    $BIOASQ_DIR/6B1_golden.json \
    RESULTS_PATH/BioASQform_BioASQ-answer.json
```
As our example is on factoid questions, the result will be like
``` 
0.0 0.4358974358974359 0.6153846153846154 0.5072649572649572 0.0 0.0 0.0 0.0 0.0 0.0
```
where the second, third and fourth numbers will be SAcc, LAcc and MRR of factoid questions respectively.

Please be advised that the performance of yesno questions has relatively high variance. 
Following is our result of five independent experiments on yesno (6b) questions (We used settings of `Snippet as-is` dataset, `BioBERT v1.1 - SQuAD v2.0` model. Please see [examplecode.sh](examplecode.sh) for details.).


|          |  1st  |  2nd  |  3rd  |  4th  |  5th  | Average |
|----------|-------|-------|-------|-------|-------|---------|
| Macro F1 | 74.11 | 78.46 | 80.89 | 71.57 | 81.25 | **77.256**  |


**Be sure to clean `output_dir` in order to perform independent experiments. Otherwise, our code will skip training and reuse existing model in the `output_dir` for prediction**

## Requirement
* GPU (Our setting was Titan Xp with 12Gb graphic memory)
* Python 3 (Not working on python 2; encoding issues for run_yesno.py)
* TensorFlow v1.11 (Not working on TF v2)
* For other software requirement details, please check `requirements.txt` 

## License and Disclaimer
Please see and agree `LICENSE` file for details. Downloading data indicates your acceptance of our disclaimer.


## Contact information

For help or issues using our model, please contact Wonjin Yoon (`wonjin.info {at} gmail.com`) for communication related to the paper.
<br>We welcome any suggestion regarding this repository.
<br>**Please denote the name of our paper when you contact me (Wonjin)** since I maintain BioBERT and other repositories using BioBERT.
