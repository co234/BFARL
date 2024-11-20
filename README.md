# acml-in-processing
This repository conatins the implementation of the paper "Bias-Tolerant Fair Classification," accepted at ACML 2021.

## Abstract
The label bias and selection bias are acknowledged as two reasons in data that will hinder the fairness of machine-learning outcomes. The label bias occurs when the labeling decision is disturbed by sensitive features, while the selection bias occurs when subjective bias exists during the data sampling. Even worse, models trained on such data can inherit or even intensify the discrimination. Most algorithmic fairness approaches perform an empirical risk minimization with predefined fairness constraints, which tends to trade-off accuracy for fairness. However, such methods would achieve the desired fairness level with the sacrifice of the benefits (receive positive outcomes) for individuals affected by the bias. Therefore, we propose a **B**ias-Toleran **FA**ir **R**egularized **L**oss (B-FARL), which tries to regain the benefits using data affected by label bias and selection bias. B-FARL takes the biased data as input, calls a model that approximates the one trained with fair but latent data, and thus prevents discrimination without constraints required. In addition, we show the effective components by decomposing B-FARL, and we utilize the meta-learning framework for the B-FARL optimization. The experimental results on real-world datasets show that our method is empirically effective in improving fairness towards the direction of true but latent labels.


## Reference
```
@InProceedings{pmlr-v157-zhang21d,
  title = 	 {Bias-tolerant Fair Classification},
  author =       {Zhang, Yixuan and Zhou, Feng and Li, Zhidong and Wang, Yang and Chen, Fang},
  booktitle = 	 {Proceedings of The 13th Asian Conference on Machine Learning},
  pages = 	 {840--855},
  year = 	 {2021},
  editor = 	 {Balasubramanian, Vineeth N. and Tsang, Ivor},
  volume = 	 {157},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--19 Nov},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v157/zhang21d/zhang21d.pdf},
  url = 	 {https://proceedings.mlr.press/v157/zhang21d.html},
}
```
