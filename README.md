# allRank : Learning to Rank in PyTorch

### Rubric

- Quality of content(30%)
- Interview(25%)
- Writing(10%)
- Peer assessment(5%)

### Report Requirement

The report should include the following:

- Title, group id, authors.
- Abstract. Ideally a paragraph long
  - It’s good to give some background for the work right at the beginning, define the current
    proposal and situate it in that background, summarize the main findings, and end by
    identifying the wider significance of the work. The “General reasoning” section of your
    experiment protocol is likely to provide good content for the abstract. Remember that an
    Abstract is not a summarized Introduction. Instead, it is a higher-level description of your
    work.
- Introduction. Problem statement, overall plan to tackle the problem.
  - This is a particularly important part of the paper. In this section, the reader is likely to
    form their expectations for the work and start to form their opinions of it. The
    introduction should tell the whole story of the paper, in a way that is understandable to
    most people in the field:
    - Where are we? That is, what area of the field are you working in? Answering this
      question is important for orienting the reader.
    - What is the goal of the work presented in this paper? If it is a new
      method/solution, that often includes a hypothesis. If it is an empirical analysis,
      that instead generally refers to the need for comparing/contrasting.
    - What concepts does your work rely on? You can’t expect your reader to fill in the
      gaps. When necessary, define terminology as part of your narrative.
    - What are the main findings of the paper?
  - By the end of the Introduction, the reader should be able to know what is the problem of
    interest, what makes it a problem, what did you propose to do and why, and what are the
    main contributions/lessons learned emerging from this work.
- Related work. This section can be relatively short. A good strategy for this section is to first group
  the papers you want to cover into general categories that relate to your own work in important
  ways. For each such category, state its thematic unity, briefly discuss what each paper
  accomplishes, and then, importantly, relate this work to your own project, as a way of providing
  background for your work and distinguishing it from previous work. In this way, you create a space
  for your own contribution.
- Method. Length highly variable.
  - If you are discussing an empirical analysis, then this is the section where you describe the
    framework for your analysis, including, but not limited to data, metrics, procedures, and
    methods/solutions under analysis.
  - If you are discussing a new method/solution, then this is the section where you describe
    the technical details and design decisions of your proposed method/solution.
- Experiments. Length highly variable. error analysis.
  - If you are discussing an empirical analysis, in this section you provide an in-depth
    discussion of experiments conducted, results, and inferences that can be made from
    findings.
  - If you are discussing a new method/solution, here is where you discuss experiments
    (including data, metrics, and baselines for comparison) and results that demonstrate
    correctness/effectiveness/efficiency of your proposed method/solution.
  - Regardless of the focus of your work, in this section you should also discuss limitations.
- Conclusions. describe what you learned/found and what avenues for further improvement you
  see.
  - It's best to briefly summarize what the paper did and why, and then try to articulate the
    broader significance of the work, looking ahead to expanding its scope.
- Self-Assessment of Contributions. Each student should briefly outline their contribution to this
  project.
  - Here you can discuss implementation responsibilities, writing, data collection,
    management, etc.
- References.

### Motivation

allRank provides an easy and flexible way to experiment with various LTR neural network models and loss functions.
It is easy to add a custom loss, and to configure the model and the training procedure. 
We hope that allRank will facilitate both research in neural LTR and its industrial applications.

## Features

### Implemented loss functions:
 1. RankNet
 2. ApproxNDCG
 3. RMSE
 4. NeuralNDCG (introduced in https://arxiv.org/pdf/2102.07831)

### Getting started guide

Download dataset and put it into data folder(you need to create that one in the root folder)

first run

```shell
pip install -r requirements.txt
```

after that run pipeline_test.


## Research

This framework was developed to support the research project [Context-Aware Learning to Rank with Self-Attention](https://arxiv.org/abs/2005.10084). If you use allRank in your research, please cite:
```
@article{Pobrotyn2020ContextAwareLT,
  title={Context-Aware Learning to Rank with Self-Attention},
  author={Przemyslaw Pobrotyn and Tomasz Bartczak and Mikolaj Synowiec and Radoslaw Bialobrzeski and Jaroslaw Bojar},
  journal={ArXiv},
  year={2020},
  volume={abs/2005.10084}
}
```
Additionally, if you use the NeuralNDCG loss function, please cite the corresponding work, [NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting](https://arxiv.org/abs/2102.07831):
```
@article{Pobrotyn2021NeuralNDCG,
  title={NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting},
  author={Przemyslaw Pobrotyn and Radoslaw Bialobrzeski},
  journal={ArXiv},
  year={2021},
  volume={abs/2102.07831}
}
```

## License

Apache 2 License
