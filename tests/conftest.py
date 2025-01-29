from pathlib import Path

import pytest


@pytest.fixture
def rootdir(request):
    return Path(request.config.rootdir)


@pytest.fixture
def use_cases_dir():
    use_cases_dir = Path(__file__).parent / '../autoqml_lib/use_cases/'
    return use_cases_dir
