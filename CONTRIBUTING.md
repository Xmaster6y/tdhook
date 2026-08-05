# How to Contribute?

TDHook uses [`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/).

## Dev Install

Install the dependencies and the pre-commit hooks:

```bash
just install
```

## Checks

```bash
just checks
just tests
```

Run only the executable demo notebooks with:

```bash
just notebook-tests
```

CI notebooks must be deterministic, CPU-friendly, complete in under two
minutes, and avoid network access. Mark them with
`metadata.tdhook.ci = true`.

## Branches

Make a branch in your fork before making a pull request to `main`.

## Submitting Ideas

Ideas can be submitted through the [GitHub Discussions](https://github.com/Xmaster6y/tdhook/discussions) or via [Roadmap Issues](https://github.com/Xmaster6y/tdhook/issues/new?&template=roadmap.yml).
