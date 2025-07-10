# In Between Sim

## Local Installation

If you have `uv` installed:

```shell
uv sync
```

If not:

```shell
pip install -r requirements.txt
```

## Running

If you have `uv` installed:

```shell
uv run main.py
```

If not:

```shell
python main.py
```

## Tensorboard

See the training mean reward and mean length with Tensorboard.

If you have `uv` installed:

```shell
uv run tensorboard --logdir=logs
```

If not:

```shell
python -m tensorboard --logdir=logs
```
