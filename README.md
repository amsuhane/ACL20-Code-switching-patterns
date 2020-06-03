# Code-switching patterns can be an effective route to improve performance of downstream NLP applications: A case study of humour, sarcasm and hate speech detection [link](https://arxiv.org/abs/2005.02295)
## How to run:

**Humour**
To run ML model:
- Baseline: python  Humour/ML/grid_search_baseline.py
- Switching: python  Humour/ML/grid_search_baseline_switching.py
To run HAN:
- Baseline: python Humour/HAN/master_script_baseline_signal.py
- Switching: python Humour/HAN/master_script_switching_signal.py

**Hate**
To run ML model:
- Baseline & Switching: python  Hate/ML/grid_search.py
To run HAN:
- Baseline: python Hate/HAN/grid_search_baseline.py
- Switching: python Hate/HAN/grid_search_switching.py

**Sarcasm**
To run ML model:
- Baseline: python Sarcasm/ML/Baseline/classification.py
- Switching: python Sarcasm/ML/Switching/classification.py
To run HAN:
- Baseline: python Sarcasm/HAN/grid_search_baseline.py
- Switching: python Sarcasm/HAN/grid_search_switching.py
