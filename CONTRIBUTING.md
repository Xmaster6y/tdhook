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

## Installed-package checks

CI and publishing share the `Package` workflow. It builds the wheel and source
distribution once, then installs them into fresh environments without `uv.lock`
or development dependencies. CPU smoke checks cover the wheel on Python
3.11–3.13 with newly resolved dependencies, the wheel with the declared minimum
PyTorch/TensorDict versions on Python 3.11, and the source distribution on
Python 3.11. Update the minimum-version matrix when changing dependency bounds.

The checks execute the README example, repeated activation captures, and a
capture-to-analysis workflow outside the checkout, with Python's isolated mode
enabled. They verify installed-package provenance, numerical results, and hook
cleanup. Publishing waits for every check and uploads those same distributions.

To check a wheel locally with newly resolved dependencies:

```bash
just build
just release-smoke dist/tdhook-0.2.1-py3-none-any.whl 0.2.1
```

Use the version and wheel path produced by your build. The local command also
isolates imports from the checkout; the CI checks explicitly use CPU PyTorch.

## Branches

Make a branch in your fork before making a pull request to `main`.

## Submitting Ideas

Ideas can be submitted through the [GitHub Discussions](https://github.com/Xmaster6y/tdhook/discussions) or via [Roadmap Issues](https://github.com/Xmaster6y/tdhook/issues/new?&template=roadmap.yml).
