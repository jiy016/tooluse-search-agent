# Search-o1 testing and analysis

This repository contains our implementation and evaluation of the **Search-o1** framework, which is a search-enhanced, multi-step reasoning agent.  We used and modified a version of the original codebase and evaluated it on a small HotpotQA subset already included in this repository, and ready to run and test. The following are a brief explanation on how to run, use, and interpret results.



## Introduction
**Search-o1** is a reasoning framework that enhances LLMs by integrating:
- web search  
- document inspection  
- multi-step reasoning  
- search queries  

Unlike standard RAG systems, Search-o1:
- searches **multiple times**
- refines queries
- extracts relevant evidence
- updates its reasoning at each step

This repo uses UCSD's DSMLP-cluster implementation and our modifications to the search backend we used Serper.dev instead of Bing.


## Repository Structure

-  `scripts/` is where  `run_search_o1.py` is stored, which handles query generation, retrieval, and model reasoning.  
- The `data/QA_Datasets` is where our data is stored under the file name `hotpotqa.json` 
- The `models/` directory defines model loading and inference utilities for running **Qwen** or any other supported LLM.   
- All produced outputs—such as **JSONL logs** of model thoughts, search calls, and final answers—are automatically stored in the `outputs/` directory for each dataset split.  

Together, this structure provides a clean separation between code, configuration, and generated results so you can easily navigate and modify any part of the system.
If you



## Instructions 






## 📄 Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{Search-o1,
  author       = {Xiaoxi Li and
                  Guanting Dong and
                  Jiajie Jin and
                  Yuyao Zhang and
                  Yujia Zhou and
                  Yutao Zhu and
                  Peitian Zhang and
                  Zhicheng Dou},
  title        = {Search-o1: Agentic Search-Enhanced Large Reasoning Models},
  journal      = {CoRR},
  volume       = {abs/2501.05366},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2501.05366},
  doi          = {10.48550/ARXIV.2501.05366},
  eprinttype    = {arXiv},
  eprint       = {2501.05366},
  timestamp    = {Wed, 19 Feb 2025 21:19:08 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2501-05366.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```



## LICENSE 

