import mutafitup.datasets as datasets_module
from mutafitup.datasets import (
    BaseMultitaskDataset,
    DataCollator,
    PerProteinClassificationDataset,
    PerProteinRegressionDataset,
    PerResidueClassificationDataset,
    PerResidueRegressionDataset,
)


def test_datasets_exports_match_all():
    expected = {
        "BaseMultitaskDataset": BaseMultitaskDataset,
        "DataCollator": DataCollator,
        "PerProteinClassificationDataset": PerProteinClassificationDataset,
        "PerProteinRegressionDataset": PerProteinRegressionDataset,
        "PerResidueClassificationDataset": PerResidueClassificationDataset,
        "PerResidueRegressionDataset": PerResidueRegressionDataset,
    }

    assert sorted(datasets_module.__all__) == sorted(expected.keys())
    for name, obj in expected.items():
        assert getattr(datasets_module, name) is obj
