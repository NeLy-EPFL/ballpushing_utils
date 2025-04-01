# Tests whether main ball-pushing objects are created correctly
import pytest


@pytest.mark.parametrize("fixture_name, expected_type", [("example_fly", "Fly"), ("example_experiment", "Experiment")])
def test_object_creation(request, fixture_name, expected_type):
    obj = request.getfixturevalue(fixture_name)
    assert obj.__class__.__name__ == expected_type


def test_fly_properties(example_fly):
    # Check required attributes for the Fly class
    required_attrs = ["directory", "experiment", "config", "metadata", "Genotype"]
    for attr in required_attrs:
        assert hasattr(example_fly, attr), f"Fly missing {attr}"
    assert example_fly.directory.exists(), "Fly directory path is invalid"
    assert example_fly.metadata is not None, "Fly metadata is missing"
    assert example_fly.Genotype is not None, "Fly Genotype is missing"


def test_experiment_structure(example_experiment):
    # Check required attributes for the Experiment class
    required_attrs = ["directory", "config", "metadata", "flies", "fps"]
    for attr in required_attrs:
        assert hasattr(example_experiment, attr), f"Experiment missing {attr}"
    assert example_experiment.directory.exists(), "Experiment directory path is invalid"
    assert isinstance(example_experiment.metadata, dict), "Experiment metadata is not a dictionary"
    assert len(example_experiment.flies) > 0, "No flies loaded in the experiment"
    assert isinstance(example_experiment.fps, (int, float)), "Experiment FPS is not a valid number"
