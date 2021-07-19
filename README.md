![HEAR2021](https://neuralaudio.ai/assets/img/hear-header-sponsor.jpg)
# HEAR 2021 Submission Validator

This package provides a command-line tool to verify that a python module follows the
HEAR 2021 [common API](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api).

For full details on the HEAR 2021 NeurIPS competition please visit the
[competition website.](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html)

### Installation
```python
pip install hearvalidator
```
This will install a command-line tool: `hear-validator`

### Usage
```python
hear-validator <module-to-test> --model <path-to-model-checkpoint-file> --device <device-to-run-on>
```
If the `device` isn't specified then this will run on a GPU if one is present. If no
GPU is available then this will default to the CPU.
##### Example usage:
```python
hear-validator hearbaseline --model /path/to/baseline-weights --device cuda
```
