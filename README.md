# LEA: Improving Sentence Similarity Robustness to Typos Using Lexical Attention Bias (KDD'23)

### Main research track @ [29th ACM Conference on Knowledge Discovery and Data Mining (KDD'23)](https://kdd.org/kdd2023/)

Authors: Mario Almagro*, Emilio Almazán*, Diego Ortego, David Jiménez*

Affiliation: [NIQ](https://nielseniq.com/global/en/) Innovation (Core Models team)

Link to the paper: [arxiv](https://arxiv.org/pdf/2307.02912.pdf).

## NEWS!

- [December 19, 2023] LEA4BERT code available.
- [December 18, 2023] Added link to download all test splits in the paper.
    - Link to the datasets: [zenodo](https://zenodo.org/records/10401846).
- [December 18, 2023] Updated documentation.

## Pending tasks

* [x] Update documentation.
* [x] Upload test splits.
* [ ] Upload LEA's module code.

## Abstract
Textual noise, such as typos or abbreviations, is a well-known issue that penalizes vanilla Transformers for most downstream tasks. We show that this is also the case for sentence similarity, a fundamental task in multiple domains, e.g. matching, retrieval or paraphrasing. Sentence similarity can be approached using cross-encoders, where the two sentences are concatenated in the input allowing the model to exploit the inter-relations between them. Previous works addressing the noise issue mainly rely on data augmentation strategies, showing improved robustness when dealing with corrupted samples that are similar to the ones used for training. However, all these methods still suffer from the token distribution shift induced by typos. In this work, we propose to tackle textual noise by equipping cross-encoders with a novel LExical-aware Attention module (LEA) that incorporates lexical similarities between words in both sentences. By using raw text similarities, our approach avoids the tokenization shift problem obtaining improved robustness. We demonstrate that the attention bias introduced by LEA helps cross-encoders to tackle complex scenarios with textual noise, specially in domains with short-text descriptions and limited context. Experiments using three popular Transformer encoders in five e-commerce datasets for product matching show that LEA consistently boosts performance under the presence of noise, while remaining competitive on the original (clean) splits. We also evaluate our approach in two datasets for textual entailment and paraphrasing showing that LEA is robust to typos in domains with longer sentences and more natural context. Additionally, we thoroughly analyze several design choices in our approach, providing insights about the impact of the decisions made and fostering future research in cross-encoders dealing with typos.


Figure 1 shows an illustrative real example of how LEA's attention bias provides higher robustness under the presence of textual noise (best view with white background - no dark mode friendly).

![LEA_diagram](/images/graphical_abstract_LEA.png)

*Figure 1: Influence of LEA's lexical bias in the overall attention of one sample in Abt-Buy (BERT-medium cross-encoder). We show the attention of one word on the left sentence (black) to all tokens on the right one. Note that the stronger the red color is, the higher the attention*

In Figure 2 we show how to integrate LEA within the attention module of regular transformers (best view with white background - no dark mode friendly).

![LEA_diagram](/images/LEAdiagram.png) 

*Figure 2: Overview of the attention mechanism in Transformers where we add the proposed lexical attention bias (LEA). We use the traditional nomenclature for the key, query and value representations (Q, K, V).*

## Test splits

LEA has been tested in several datasets using a corrupted version of their test splits. In this [LINK](https://zenodo.org/records/10401846) you can download the test splits for all datasets used in the paper. 

As described in the paper, we introduce textual noise that is normally present in the domains of each dataset. Training and validation splits remain the same as the original datasets.

Available datasets:
- WDC-Computers (small, medium, large, xlarge)
- WDC-All (xlarge)
- Abt-Buy
- Amazon-Google
- RTE
- MRPC

## Instructions to reproduce results

In this section we will provide the instructions to reproduce the results of the paper. Note that the code shared in this repo is extracted from a private library, so one might expect minor changes on the results.

## Main results

- E-commerce public datasets:
<img src="images/LEA_results_ecom_noise.jpg" height="280rm">

- Natural language public datasets:
<img src="images/LEA_results_natural_noise.jpg" height="150rm">

- Comparison with auto-regressive models of comparable sizes:
<img src="images/LEA_results_gpt_noise.jpg" height="300rm">

## Citation

```
@inproceedings{almagro2023lea,
  title={LEA: Improving Sentence Similarity Robustness to Typos Using Lexical Attention Bias},
  author={Almagro, Mario and Almaz{\'a}n, Emilio and Ortego, Diego and Jim{\'e}nez, David},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={36--46},
  year={2023}
}
```
