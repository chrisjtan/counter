## Overall
Pytorch implementation for paper 
"Counterfactual Explainable Recommendation".

![](pic/overview.png)
### Paper link: 
https://arxiv.org/abs/2108.10539

## Requirements
- Python 3.7
- pytorch 1.1.0
- cuda 9

## Instruction
1. Before running the experiments, please set the "--review_dir" and "--sentires_dir" arguments to the paths of the review dataset and extracted sentiment dataset. We provide default argument files in the /utils folder.\
You may download Amazon Review dataset from https://jmcauley.ucsd.edu/data/amazon/ and Yelp Review dataset from https://www.yelp.com/dataset.
2. The sentiment data are extracted with "Sentires" tool https://github.com/evison/Sentires. A python guide can be found in https://github.com/lileipisces/Sentires-Guide. You can also use any linguistic tool to extract such data. 
3. We provide an example on "Cell Phones and Accessories" datasets. The pre-extracted sentiment data is already in the dataset/Cell_Phones_and_Accessories" folder, but you have to download the review dataset by yourself due to github size limit.
4. Under the project root folder, run:
    ```
    source setup.sh
    ```
5. Training the base recommender: run:
    ```
    python scripts/train_base_amazon.py
    ```
6. For generating explanations, run:
    ```
    python scripts/generate_exp_amazon.py
    ```
