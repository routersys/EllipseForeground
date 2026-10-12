from harness.cases import all_cases
from harness.golden import GOLDEN_PATH, Golden


def test_case_names_are_unique():
    names = [case.name for case in all_cases()]
    assert len(names) == len(set(names))


def test_committed_golden_lists_every_case():
    assert [case.name for case in Golden.load().cases] == [case.name for case in all_cases()]


def test_the_golden_file_is_committed_next_to_the_harness():
    assert GOLDEN_PATH.exists()
