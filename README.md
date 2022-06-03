# HEAR Benchmark Submission Validator

This package provides a command-line tool to verify that a python module follows the
HEAR [common API](https://hearbenchmark.com/hear-api.html).

For full details on the HEAR benchmark please visit https://hearbenchmark.com

### Installation

Tested with Python 3.7 and 3.8. Python 3.9 is not officially supported
because pip3 installs are very finicky, but it might work.

```python
pip install hearvalidator
```
This will install a command-line tool: `hear-validator`

### Usage

Let's validate the [HEAR naive baseline model](https://github.com/hearbenchmark/hear-baseline):
```
pip install hearbaseline
wget https://github.com/hearbenchmark/hear-baseline/raw/main/saved_models/naive_baseline.pt
hear-validator hearbaseline --model ./naive_baseline.pt
```

Optional arguments for `hear-validator`:

 * `--model`: Load the model weights from this path.
 * `--device`: Device to run validation on. If this isn't specified then this
 will run on a GPU if one is present. If no GPU is available then this will
 default to the CPU.
