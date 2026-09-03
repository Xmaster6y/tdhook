"""Keep the public getting-started examples aligned and executable."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _markdown_python_block(text: str) -> str:
    start = text.index("```python\n") + len("```python\n")
    end = text.index("\n```", start)
    return text[start:end]


def _rst_python_blocks(text: str) -> tuple[str, ...]:
    lines = text.splitlines()
    blocks = []
    for directive, line in enumerate(lines):
        if line != ".. code-block:: python":
            continue
        block = []
        for content in lines[directive + 2 :]:
            if content.startswith("   "):
                block.append(content[3:])
            elif not content:
                block.append("")
            else:
                break
        blocks.append("\n".join(block).rstrip())
    return tuple(blocks)


def test_getting_started_examples_match_and_execute():
    readme_example = _markdown_python_block((REPO_ROOT / "README.md").read_text())
    docs_example = _rst_python_blocks((REPO_ROOT / "docs/source/start.rst").read_text())[0]

    assert docs_example == readme_example

    namespace = {}
    exec(docs_example, namespace)

    assert tuple(namespace["attributions"].shape) == (1, 4)
    assert all(not module._forward_hooks for module in namespace["model"].modules())


def test_getting_started_workflow_example_executes_after_the_first_example():
    blocks = _rst_python_blocks((REPO_ROOT / "docs/source/start.rst").read_text())
    assert len(blocks) == 2
    namespace = {}

    exec(blocks[0], namespace)
    exec(blocks[1], namespace)

    result = namespace["result"]
    assert ("attr", "input") in result
    assert "attribution_mass" in result
