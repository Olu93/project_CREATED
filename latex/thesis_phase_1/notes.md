# Introduction

- Introduce State of the art
- Introduce the limitations

- [ ] Motivation
    - Either motivate the topic and then describe the research question
    - OR state the research question and then motivate the reason for why the question is relevant

- [ ] Research Question
- [ ] General Approach
    - Less detailed version of the concept visual

- [ ] What is Process Mining?
    - [ ] Types of Processes
        - Continuous \& Discrete 
        - Finite \& Infinite
    - [ ] Mathematical properties as stochastic process: shorturl.at/huC18
        - Index set, state space and sample function
        - BProcesses are non-stationary - Depending on the timestep different distribution
            - No Granger causality. There's no periodicity in the data. Distributions may change at branches.
            - In reality there's a concept drift
            - Stationarity is often assumed in research
        - Ergodicity
            - Non ergodic
        - Finite dimensional
        - Non-markovian (Could be made markovian)
        - Martingale as process stops at some point and stays the same
        - https://en.wikipedia.org/wiki/Sequential\_pattern\_mining
        - Degrees of seperation: shorturl.at/qxIVY
    - [ ] Challenges of Processing Process Data
        - [ ] Structural and Statistical Features of Multivariate Sequences
            - [ ] Structural
                - Structural features like variable sequence length
                - Multivariate Timeseries Causality of Sequential data steps
            - [ ] Statistical
                - Curse of dimensionality 
                - Mixed Data processing
                    - Continuization: https://stats.stackexchange.com/a/28228
                    - Discretization: https://stats.stackexchange.com/a/28228
                    - Mixed Variate: https://stats.stackexchange.com/a/28227
                        - Mahanobolis Distance
                        - Shortest Path
                        - Linear Combination (HEOM): shorturl.at/flqEG
                        - Gower: 
                        - etc.: Bishnoi, S., \& Hooda, B. K. (2020)
        - [ ] Nature of Processes
            - Imbalance of shorter sequences vs longer sequences
                - Use combines histogram of all datasets as example
            - Sequence cycles
            - Parallel sequences
            - Branching factors per step "Curse of multiplicity"
            - Concept Drift (Processes may change and that changes the distribution which makes it non-stationary)

    - Process might not have to be explained extensively. Label transition system
    - It could also be described as a stochastic Language with a grammar and the sequences are the realisations

# Methods


- [ ] Datasets

    - [ ] report log size, 
    - [ ] percentage of distinct traces, 
    - [ ] sequence lengths
    - [ ] number of event 
    - [ ] show example of the original trace


- [ ] Preprocessing


- [ ] Framework
    - Detailed version of the concept visual


# Results


# Evaluation

- [ ] Practical considerations
    - Limiting the evaluation sequence length to true sequence length only
- [ ] Metrics
    - List all ( https://en.wikipedia.org/wiki/Edit\_distance )
        - Explain why not Hamming 
        - Explain why not plain Levenshtein (Verenich et. al.)
        - Add metric overview image from (Wang, J., \& Dong, Y. (2020). Measurement of Text Similarity: A Survey. Information, 11(9), 421. https://doi.org/10.3390/info11090421)
    - Discuss distance mathematical distance properties
    - Discuss distance characteristics
        - Local alignment, Global Alignment, Parallelism, Focus on start



# Discussion

- [ ] Two perspectives of this work
    - Application faithfullness means model needs to be 100\% accurate
    - If model is not accurate the work is just an explainer for the model


# Conclusion
