# Compile by Training

Turn a natural-language function description into a reusable neural program.

Compile by Training uses teacher models to generate examples, finetunes a small PAW interpreter, and produces a `.paw` function that runs locally without further teacher calls.

Requires Python 3.10+, [`uv`](https://docs.astral.sh/uv/), and an OpenAI API key. The default recipe is intended for an accelerator with about 40 GB of available memory; use a lower `--micro-batch-size` or `--gradient-checkpointing` on smaller devices.

## Compile

Set an OpenAI API key and run the script:

```bash
export OPENAI_API_KEY=...
uv run compile.py "Classify sentiment. Return only positive, negative, or neutral." -o sentiment.paw
```

The script saves `sentiment.paw`, installs it in the local PAW cache, and prints its program ID. Use the ID with the PAW Python package:

```bash
pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

```python
import programasweights as paw

sentiment = paw.function("<program-id>")
print(sentiment("I loved it."))
```

The program downloads the shared 0.6B interpreter once, then runs locally. A PAW API key is optional; setting `PAW_API_KEY` provides higher hosted compile limits.

Mapper assets and completed teacher synthesis are cached, so rerunning the same specification and recipe reuses finished work.

## Customize

The defaults reproduce the released Compile by Training recipe. Every major choice can be changed from the command line:

```bash
uv run compile.py "Classify sentiment. Return only positive or negative." \
  --teacher gpt-5.4-mini=800 \
  --teacher gpt-5.5=400 \
  --steps 80 \
  --batch-size 32 \
  --micro-batch-size 8 \
  --learning-rate 1e-4 \
  -o sentiment.paw
```

```bash
uv run compile.py --print-config
uv run compile.py --help
```

## How it works

1. Compile the specification once with the fast PAW compiler to initialize the neural program.
2. Ask teacher models to generate task-specific input-output examples.
3. Finetune the local interpreter on those examples.
4. Package the result as a reusable `.paw` function.

The implementation is contained in [`compile.py`](compile.py).

## Citation

If you use this compiler in research, please cite Compile by Training for the training-based compiler and Program-as-Weights for the underlying fuzzy-function programming model.

**[Compile by Training: Turning Natural-Language Specifications into Local Neural Functions](https://arxiv.org/abs/2609.04199)**

```bibtex
@inproceedings{paw_2609_04199,
  title = {{Compile by Training: Turning Natural-Language Specifications into Local Neural Functions}},
  author = {Deng, Yuntian and Nie, Pengyu and Shieber, Stuart},
  booktitle = {EMNLP 2026 System Demonstrations},
  year = {2026},
  url = {https://arxiv.org/abs/2609.04199}
}
```

**[Program-as-Weights: A Programming Paradigm for Fuzzy Functions](https://arxiv.org/abs/2607.02512)**

NeurIPS 2026 · [Preprint](https://arxiv.org/abs/2607.02512)

```bibtex
@inproceedings{zhang2026programasweights,
  title = {{Program-as-Weights: A Programming Paradigm for Fuzzy Functions}},
  author = {Zhang, Wentao and Hotsko, Liliana and Kim, Woojeong and Nie, Pengyu and Shieber, Stuart and Deng, Yuntian},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year = {2026},
  url = {https://arxiv.org/abs/2607.02512}
}
```
