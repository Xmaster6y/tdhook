# How to Contribute?

## Guidelines

The project dependencies are managed using `uv`, see their installation [guide](https://docs.astral.sh/uv/).

Additionally, to make your life easier, install `just` to use the shortcut commands.

## Dev Install

Install the dependencies and the pre-commit hooks:

```bash
just install
```

## Ensuring CI Passes

To ensure CI passes, run the checks and tests. To run the checks (`pre-commit` checks):

```bash
just checks
```

To run the tests (using `pytest`):

```bash
just tests
```

## Branches

Make a branch in your fork before making a pull request to `main`.

## Submitting Ideas

Ideas can be submitted through the [GitHub Discussions](https://github.com/Xmaster6y/tdhook/discussions) or via [Roadmap Issues](https://github.com/Xmaster6y/tdhook/issues/new?&template=roadmap.yml).
