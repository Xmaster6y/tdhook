"""Keep the two public getting-started examples aligned and executable."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _markdown_python_block(text: str) -> str:
    start = text.index("```python\n") + len("```python\n")
    end = text.index("\n```", start)
    return text[start:end]


def _rst_python_block(text: str) -> str:
    lines = text.splitlines()
    directive = lines.index(".. code-block:: python")
    block = []
    for line in lines[directive + 2 :]:
        if line.startswith("   "):
            block.append(line[3:])
        elif not line:
            block.append("")
        else:
            break
    return "\n".join(block).rstrip()


def test_getting_started_examples_match_and_execute():
    readme_example = _markdown_python_block((REPO_ROOT / "README.md").read_text())
    docs_example = _rst_python_block((REPO_ROOT / "docs/source/start.rst").read_text())

    assert docs_example == readme_example

    namespace = {}
    exec(docs_example, namespace)

    assert tuple(namespace["attributions"].shape) == (1, 4)
    assert all(not module._forward_hooks for module in namespace["model"].modules())
