from experiments.residual_screen.residual_experiment import SpringResidualDataset


def test_batched_items_preserve_order():
    dataset = SpringResidualDataset.__new__(SpringResidualDataset)
    dataset.__getitem__ = lambda index: index * 2

    assert dataset.__getitems__([3, 1, 2]) == [6, 2, 4]
    assert dataset.__getitems__([]) == []
