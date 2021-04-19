# Code-switching patterns can be an effective route to improve performance of downstream NLP applications: A case study of humour, sarcasm and hate speech detection 

Here we have the code and data for the following paper: [Code-switching patterns can be an effective route to improve performance of downstream NLP applications: A case study of humour, sarcasm and hate speech detection](https://arxiv.org/abs/2005.02295) by Srijan Bansal, Vishal Garimella, Ayush Suhane, Jasabanta Patro, Animesh Mukherjee. Proceedings of ACL 2020

**Our trained embedding:**
You can download the embedding [here](https://bit.ly/3dtfxDd)

## How to run:

#### Humour
ML model:
- Baseline: run `python  grid_search_baseline.py` from Humour/ML/
- Switching: run `python  grid_search_baseline_switching.py` from Humour/ML/

To run HAN:
- Baseline: run `python master_script_baseline_signal.py` from Humour/HAN/
- Switching: run `python master_script_switching_signal.py` from Humour/HAN/

#### Hate
To run ML model:
- Baseline & Switching: run `python  grid_search.py` from Hate/ML/

To run HAN:
- Baseline: run `python grid_search_baseline.py` from Hate/HAN/
- Switching: run `python grid_search_switching.py` from Hate/HAN/

#### Sarcasm
To run ML model:
- Baseline: run `python classification.py` from Sarcasm/ML/Baseline/
- Switching: run `python classification.py` from Sarcasm/ML/Switching/

To run HAN:
- Baseline: run `python grid_search_baseline.py` from Sarcasm/HAN/
- Switching: run `python grid_search_switching.py` from Sarcasm/HAN/
