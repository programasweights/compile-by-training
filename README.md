# Compile by Training

Turn a natural-language function description into a reusable neural program.

Compile by Training uses teacher models to generate examples, finetunes a small PAW interpreter, and produces a `.paw` function that runs locally without further teacher calls.

## Compile

Set an OpenAI API key and run the script with [`uv`](https://docs.astral.sh/uv/):

```bash
export OPENAI_API_KEY=...
uv run compile.py "Classify sentiment. Return only positive, negative, or neutral." -o sentiment.paw
```

The script prints a program ID. Use it with the PAW Python package:

```bash
pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

```python
import programasweights as paw

sentiment = paw.function("<program-id>")
print(sentiment("I loved it."))
```

The program downloads the shared 0.6B interpreter once, then runs locally. A PAW API key is optional; setting `PAW_API_KEY` provides higher hosted compile limits.

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

## Paper

[Compile by Training: Turning Natural-Language Specifications into Local Neural Functions](https://arxiv.org/abs/2609.04199), Yuntian Deng, Pengyu Nie, and Stuart Shieber.

## License

MIT
