import pytest

from harness.cases import CaseContext
from harness.golden import Golden, current_environment
from harness.runner import transition, verify

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def context() -> CaseContext:
    return CaseContext()


def test_committed_golden_matches_the_current_output(context):
    golden = Golden.load()
    if golden.environment != current_environment():
        pytest.skip(f"基準値は別の環境で書かれている。{golden.environment}")
    assert verify(context) == 0


def test_reusing_an_analysis_gives_the_same_result_as_a_fresh_one(context, tmp_path):
    assert transition(context, tmp_path) == 0
