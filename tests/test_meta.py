from importlib.metadata import version

import tdhook


def test_version_matches_metadata() -> None:
    assert tdhook.__version__ == version("tdhook")


def test_star_import_exposes_the_documented_core_modules() -> None:
    namespace = {}

    exec("from tdhook import *", namespace)

    assert namespace["modules"] is tdhook.modules
