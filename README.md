# Psychiatric Manifestations Analysis leveraging LLM

Official implementation of the paper "Identifying Psychiatric Manifestations in Outpatients with Depression and Anxiety: A Large Language Model-Based Approach"

## Overview

<img src="doc/study_pipeline_20241127.png" alt="System Overview" width="400" height="auto">

## SMHC_psychiatry Dataset

We open source the intermediate feature sets and labels in each classification folder.

The related dialogue data and EMR data are waiting for approval under Ethics Review Board for opening to the public.

You can apply the access of the data set by email to: shihao.xu@thetahealth.ai, ianhua.chen@smhc.org.cn, xun.jiang@thetahealth.ai.

## Requirements

```

conda env create -f environment.yml
```

## Usage

### Run classification from scratch

```
python run.py
```

### To reduplicate the results in the paper, see file:

```clf_result_visualization.ipynb```.


## Citation

If you use this code, please cite:

```bibtex
@article{xu2024identifying,
  title={Identifying Psychiatric Manifestations in Outpatients with Depression and Anxiety: A Large Language Model-Based Approach},
  author={Xu, Shihao and Yan, Yiming and Li, Feng and Zhang, Shu and Ding, Yanli and Yang, Tao and Geng, Haiyang and Chen, Jianhua},
  journal={},
  year={2024}
}
```

## Contact

Welcome contact shihao.xu@thetahealth.ai for more information. 

