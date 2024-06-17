# SimStock : Representation Model for Stock Similarities

This is the origin Pytorch implementation of SimStock in the following paper: SimStock : Representation Model for Stock Similarities

![image](https://github.com/Yoontae6719/SimStock-Representation-Model-for-Stock-Similarities/assets/87846187/b5e328f2-bff7-4540-b3a0-5dac47079d17)

🚩**News**(June 16, 2024)  We will release SimStock V2 soon. **Yoontae Hwang, Stefan Zohren, Yongjea lee, 2024, Temporal Representation Learning for Stock Similarities and Its Applications in Investment Management, working paper** (SimStockV2)

🚩**News**(December 27, 2023) I'm currently preparing for my Ph.D. defense, so updates to the cleanup code are delayed. I will update it as soon as possible. 

🚩**News**(September 28, 2023): Accepted to [**ICAIF 2023**](https://ai-finance.org/icaif-23-call-for-papers/) with one strong accept and two accept!(**Aceptance rate 21%(Oral-accept)**) We will soon provide the data collection code and the cleaned-up code. (We plan to complete the work by October 14th.)
 

## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data using 1_get_dataset.ipynb.
3. Run main.py


## What is SimStock?
In this study, we introduce SimStock, a novel framework leveraging self-supervised learning and temporal domain generalization techniques to represent similarities of stock data. Our model is designed to address two critical challenges: 1) temporal distribution shift (caused by the non-stationarity of financial markets), and 2) ambiguity in conventional regional and sector classifications (due to rapid globalization and digitalization). SimStock exhibits outstanding performance in identifying similar stocks across four real-world benchmarks, encompassing thousands of stocks. The quantitative and qualitative evaluation of the proposed model compared to various baseline models indicates its potential for practical applications in stock market analysis and investment decision-making.

## Applications
The code for finding similar stocks and pairs trading has been updated. We do not provide data for Peers and indexes. However, you can purchase the data from https://site.financialmodelingprep.com/. Additionally, while we will update the code for portfolio optimization, if you wish to implement it more quickly, you can incorporate simstock embedding directly into [A Robust Co-Movement Measure for Portfolio Optimization.](https://github.com/yinsenm/gerber/tree/master)


## Citation

If you find this repo useful, please cite our paper. 

```
Will be update
```

## Contact

**I was responsible for all the code, related works, methodology, experiments, and experimental results.** If you have any questions, please feel free to contact me at **yoontae@unist.ac.kr**


## Acknowledgments

We appreciate the following github repos a lot for their valuable code base or datasets:

DRAIN : https://github.com/BaiTheBest/DRAIN
The Gerber Statistic: A Robust Co-Movement Measure for Portfolio Optimization : https://github.com/yinsenm/gerber/tree/master

## License
SPDX-FileCopyrightText: © 2023 yoontae hwang yoontae@unist.ac.kr

SPDX-License-Identifier: BSD-3-Clause

