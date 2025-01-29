# LyZNet: Examples/HSCC24

This folder contains files to run examples in the following paper:

Liu, J., Meng, Y., Fitzsimmons, M., & Zhou, R. (2024). LyZNet: A Lightweight Python Tool for Learning and Verifying Neural Lyapunov Functions and Regions of Attraction. In Proceedings of the 27th ACM International Conference on Hybrid Systems: Computation and Control (HSCC 2024). [https://doi.org/10.1145/3641513.3650134](https://doi.org/10.1145/3641513.3650134)

## Running the Examples

To run an example, execute the following command in your terminal:

```bash
python example.py
```

The expected output is provided in the corresponding `example.txt`. Please note that exact reproducibility of the results is not guaranteed due to differences in releases of dependency libraries, especially PyTorch. For more information on controlling randomness in PyTorch, visit [PyTorch Randomness](https://pytorch.org/docs/stable/notes/randomness.html). Similar results are expected, although minor variations may occur.
