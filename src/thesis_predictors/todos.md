- [x] Implement edit distance metric
    * [x] Implement Damerau-Levenstein
    * [x] Implement Smith-Watermann (Local Alignment)
    * [x] Implement Needleman-Wunsch (Global Alignment)
    * [x] Implement Jaro-Winkler (Edit-Distance with emphasis on start and only transposition)
    * [ ] Implement Time Warp Edit Distance
    * [x] Implement Sorensen Dice (Over Jaccard because of normalised domain)
    * [x] Implement Overlap coefficient (Over Jaccard because it extends it with focus on smaller sequence)
    * [x] Implement Longest Common  
    * [ ] Implement Embedding Cosine Similarity
    * [ ] Implement Embedding Eucledian Distance
- [x] Check the results
- [ ] Investigate process discovery
- [x] Implement mass int to string decoder
- [x] Implement dataset statistics 
    * [x] report log size
    * [x] percentage of distinct traces
    * [x] sequence lengths
    * [x] number of event 
    * [x] show example of the original trace
- [ ] Implement model versions with additional data
    * [x] With raw data features
    * [ ] With event embeddings
    * [ ] With trace embeddings
- [x] Add start and end token to sequence
- [x] Collect additional datasets
    * [x] Sepsis
    * [x] BPIC 2011
    * [x] BPIC 2012 
    * [x] BPIC 2013
    * [x] BPIC 2013
    * [x] BPIC 2014
    * [x] BPIC 2015
    * [x] BPIC 2016
    * [x] BPIC 2017
    * [x] BPIC 2018
    * [ ] BPIC 2019
    * [x] BPIC 2020
- [x] Implement dataset readers
- [x] Overwrite pedict function for seq2seq lstm_model
    * Take x[:-1] and x[-1] to predict y
        * x[:-1] goes through encoder to get hidden state h
        * h and x[-1] goes through encoder to retrieve prediction for next x
- [ ] Implement cross validation and randomization for runner (ds.even_splits('train', n=3))
- [x] Impl: Dataset balancing with sample weights
- [x] Impl: damerau-levenshstein tensorflow metric
- [ ] Impl: Many2Many prediction
- [ ] Impl: Extensive outcome prediction




