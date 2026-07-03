# Notebooks

The most important notebook is develop_and_compare_ms2query_2.ipynb, this notebook is fully cleaned and reproducible and contains all the main results for MS2Query 2

The other notebooks are additional analysis which are not fully cleaned and reproducible and a bit less relevant to the main story:
run_cosine_and_entropy_on_neg_neg: just also a run with cosine, but the main notebook already contains a comparison with mod cos
experiment_with_reranking_analogues: Here we did a lot of analysis on checking if we could find a reranking algorithm like the original MS2Query. During this work I figured out it is actually best to not change the MS2DeepScore prediction, but just focus on reliability prediction.
test_ann_speed_improvements: Some tests on using pynndescent for speed improvement using ANN.
