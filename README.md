# SimStock : Representation Model for Stock Similarities

This is the origin Pytorch implementation of SimStock in the following paper: SimStock : Representation Model for Stock Similarities

![image](https://github.com/Yoontae6719/SimStock-Representation-Model-for-Stock-Similarities/assets/87846187/b5e328f2-bff7-4540-b3a0-5dac47079d17)

🚩**News**(September 28, 2023): Accepted to ICAIF 2023 with one strong accept and two accept!(Aceptance rate 21%) We will soon provide the data collection code and the cleaned-up code. (We plan to complete the work by October 14th.)

🚩**News**(September 19, 2023): We will release SimStock V2 soon. **Yoontae Hwang and Yongjea lee, 2023, Universal Stock representation, working paper**
 

## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data using 1_get_dataset.ipynb.
3. Run main.py


## What is SimStock?
In this study, we introduce SimStock, a novel framework leveraging self-supervised learning and temporal domain generalization techniques to represent similarities of stock data. Our model is designed to address two critical challenges: 1) temporal distribution shift (caused by the non-stationarity of financial markets), and 2) ambiguity in conventional regional and sector classifications (due to rapid globalization and digitalization).
SimStock exhibits outstanding performance in identifying similar stocks across four real-world benchmarks, encompassing thousands of stocks. The quantitative and qualitative evaluation of the proposed model compared to various baseline models indicates its potential for practical applications in stock market analysis and investment decision-making.


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
