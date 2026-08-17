# Investigating Bias in Multilingual Language Models: Cross-Lingual Transfer of Debiasing Techniques
This GitHub repository is a fork of the official source code for *Investigating Bias in Multilingual Language Models: Cross-Lingual Transfer of Debiasing Techniques*. It was created as part of a bachelor’s thesis, in which the effectiveness of the transfer approaches across different language families is evaluated.

For this purpose, the code was modified to work with the new CrowS-Pairs dataset introduced in [*CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models*](https://aclanthology.org/2020.emnlp-main.154/) (Nangia et al., EMNLP 2020). Additionally, the repository has been restructured to provide a modular approach, allowing its components to be imported and used in Python scripts, and has been updated to support `Python 3.13`.



## Installation
```
conda create --name CrossLingualBias python=3.13
conda activate CrossLingualBias
pip install -r requirements.txt
```

## Required Datasets
The different debiasing techniques require different amounts of Wikipedia data. The datasets should be placed in the appropriate directory and can be downloaded using the links below.

| Dataset              | Download                                                                                                                           | Notes                                                                                                                        | Directory   |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------- |
| Wikipedia 2.5% ar_DZ | [Download](https://www.dropbox.com/scl/fi/5onit43s5m4326izusgk7/ar_DZ_2.5pct.txt?rlkey=qwlowff4h18qy1xdqdchhm42p&st=e8ll5rl9&dl=1) | Arabic Wikipedia Dump used for SentDebias and INLP. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/)  | `data/text` |
| Wikipedia 10% ar_DZ  | [Download](https://www.dropbox.com/scl/fi/zwyui2c8y2g63r3do6u5i/ar_DZ_10pct.txt?rlkey=a83hw2j8ar3palpizthwon2w8&st=jalgv590&dl=1)  | Arabic Wikipedia Dump used for CDA and Dropout. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/)      | `data/text` |
| Wikipedia 2.5% ca_ES | [Download](https://www.dropbox.com/scl/fi/88dl5k59s04vxfhgxt4jt/ca_ES_2.5pct.txt?rlkey=zorfuzr3uu7bhbeuu8cuuei4b&st=ffo2xlsx&dl=1) | Catalan Wikipedia Dump used for SentDebias and INLP. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/) | `data/text` |
| Wikipedia 10% ca_ES  | [Download](https://www.dropbox.com/scl/fi/dl1iisr61h2y72owgrk3u/ca_ES_10pct.txt?rlkey=w80bhul4diftlgstj86kfaovc&st=gs1a2bz8&dl=1)  | Catalan Wikipedia Dump used for CDA and Dropout. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/)     | `data/text` |
| Wikipedia 2.5% de_DE | [Download](https://www.dropbox.com/scl/fi/87ma6ezb6kvnlwm8cefpq/de_DE_2.5pct.txt?rlkey=qygifsj3zuk7pgllga7kv8rrh&st=y34ywuzr&dl=1) | German Wikipedia Dump used for SentDebias and INLP. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/)  | `data/text` |
| Wikipedia 10% de_DE  | [Download](https://www.dropbox.com/scl/fi/0fcghc4rnl1rz9qmifzjf/de_DE_10pct.txt?rlkey=w2yfr84koeajc3iuwtgd85j57&st=ts0p1hfx&dl=1)  | German Wikipedia Dump used for CDA and Dropout. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/)      | `data/text` |
| Wikipedia 2.5% en_US | [Download](https://www.dropbox.com/scl/fi/1vnp41h54dy2d6haeedb4/en_US_2.5pct.txt?rlkey=ddgzdf9v1m1izgg3uum8yqvtz&st=xfnyjivu&dl=1) | English Wikipedia Dump used for SentDebias and INLP. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/) | `data/text` |
| Wikipedia 5% en_US   | [Download](https://www.dropbox.com/scl/fi/y45xjr3w8pe71lltm7y1y/en_US_5pct.txt?rlkey=d1aq681279fi0esqvi5fp7wvq&st=u6rw273i&dl=1)   | English Wikipedia Dump used for CDA and Dropout. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/)     | `data/text` |
| Wikipedia 2.5% es_ES | [Download](https://www.dropbox.com/scl/fi/osaet3p2lgbg4btu5c9ck/es_ES_2.5pct.txt?rlkey=zz3302mtres9rp2k722qgs43q&st=j0lt6vge&dl=1) | Spanish Wikipedia Dump used for SentDebias and INLP. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/) | `data/text` |
| Wikipedia 10% es_ES  | [Download](https://www.dropbox.com/scl/fi/jh7xkpco6i1je5dgjge3m/es_ES_10pct.txt?rlkey=8d64q2jb2smw92vdizq02krsd&st=x066exa2&dl=1)  | Spanish Wikipedia Dump used for CDA and Dropout. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/)     | `data/text` |
| Wikipedia 2.5% fr_FR | [Download](https://www.dropbox.com/scl/fi/jrivdtfezj2tiqrktkzox/fr_FR_2.5pct.txt?rlkey=c8dahmw3rjhij5uinnla9j9pz&st=pbm1sthr&dl=1) | French Wikipedia Dump used for SentDebias and INLP. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/)  | `data/text` |
| Wikipedia 10% fr_FR  | [Download](https://www.dropbox.com/scl/fi/uk8q91cxya6oy587t2gv3/fr_FR_10pct.txt?rlkey=24y96v5xypqkt3ncct63oq7oz&st=i9crv6m2&dl=1)  | French Wikipedia Dump used for CDA and Dropout. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/)      | `data/text` |
| Wikipedia 2.5% it_IT | [Download](https://www.dropbox.com/scl/fi/urd8t3zew56hrwowghjed/it_IT_2.5pct.txt?rlkey=sc84diao0bamc6vb6tqcx6lx2&st=voeax8ef&dl=1) | Italian Wikipedia Dump used for SentDebias and INLP. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/) | `data/text` |
| Wikipedia 10% it_IT  | [Download](https://www.dropbox.com/scl/fi/jkqwzeuqg66ab1p9lwkwy/it_IT_10pct.txt?rlkey=xmleixjw0m8jtwwvwlah4ez1d&st=ou8ukmm5&dl=1)  | Italian Wikipedia Dump used for CDA and Dropout. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/)     | `data/text` |
| Wikipedia 100% mt_MT | [Download](https://www.dropbox.com/scl/fi/zgqoyypfkqi4e76c3bo28/mt_MT_100pct.txt?rlkey=2cdru9avwjp35ym0144en2kq5&st=4gqiqlf1&dl=1) | Maltese Wikipedia Dump used for SentDebias and INLP. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/) | `data/text` |
| Wikipedia 2.5% zh_CN | [Download](https://www.dropbox.com/scl/fi/kb24iw92x4j61gsrw0m65/zh_CN_2.5pct.txt?rlkey=8x4g2w410sgaffnltrojfquxr&st=va8xksa9&dl=1) | Chinese Wikipedia Dump used for SentDebias and INLP. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/) | `data/text` |
| Wikipedia 10% zh_CN  | [Download](https://www.dropbox.com/scl/fi/bsitzcvzubtdpbwcuqzz5/zh_CN_10pct.txt?rlkey=gut6mfbur0oydzg5kfsgdm62a&st=pz8w1pid&dl=1)  | Chinese Wikipedia Dump used for CDA and Dropout. [(Meade et al., ACL 2022)](https://aclanthology.org/2022.acl-long.132/)     | `data/text` |

> [!NOTE]
> For the English version, the 2.5% and 5% datasets were used due to memory constraints on the author's machine.

> [!NOTE]
> For the Maltese version, the 100% dataset was used because Maltese is a low-resource language.

**Comming soon...:** ~~If you want to use different splits or, for example, the full 10% of the English Wikipedia dataset, you can use the `data/text/create_corpora.py` script to generate the corpora according to your requirements.~~


## The new Modules
The new Modules can be found in `experiments/modules`. They have to be used in order to use the new languages.

> [!NOTE]
> The rest of the explanation is **coming soon**.

## Acknowledgements
This code is based on the GitHub repository of Reusens, M., Borchert, P., Mieskes, M., De Weert, J., & Baesens, B. (2023), [Investigating Bias in Multilingual Language Models: Cross-Lingual Transfer of Debiasing Techniques](https://github.com/manon-reusens/multilingual_bias). Their work, in turn, is based on Meade, N., Poole-Dayan, E., & Reddy, S. (2022, May). [An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models.](https://github.com/McGill-NLP/bias-bench/tree/main). In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 1878-1898). arXiv preprint arXiv:2110.08527.
Moreover, this code contains code of Sheng Liang, Philipp Dufter, and Hinrich Schütze. 2020. [Monolingual and Multilingual Reduction of Gender Bias in Contextualized Representations.](https://github.com/liangsheng02/densray-debiasing/tree/publish) In Proceedings of the 28th International Conference on Computational Linguistics, pages 5082–5093, Barcelona, Spain (Online). International Committee on Computational Linguistics.

We thank the authors for making their code publicly available.
