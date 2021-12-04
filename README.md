# empirical-evaluation-nyc-bail-reform-synth-control

Code for [An Empirical Evaluation of the Impact of New York's Bail Reform on Crime Using Synthetic Controls](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3964067). 

Dependencies: cvxpy, pandas, numpy, matplotlib, statsmodels 
Dependencies include [Robust Synthetic Control](https://github.com/jehangiramjad/tslib) (although primary specification is Ridge). 

1. Run ```SC_LinReg_CJA-updatedincidents.ipynb```. Change configuration cell to toggle between methods/specifications (linear regression with ridge, without ridge, RSC). ```nyc_ITS.R``` runs analysis for the ITS.
2. Generate tables: ```SC_LinReg_CJA_tables.ipynb``` reads in the outputs from each method stored in ```.p``` files in ```results``` 
