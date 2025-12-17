from importlib.metadata import version

import tdhook


def test_version_matches_metadata() -> None:
    assert tdhook.__version__ == version("tdhook")
