from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from lib.data import get_c4, get_wikitext2


class _TextDataset:
    def __init__(self, text_count=4):
        self._texts = ["local calibration text " * 8] * text_count

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, index):
        if index == "text":
            return self._texts
        if isinstance(index, slice):
            return {"text": self._texts[index]}
        return {"text": self._texts[index]}


class _Tokenizer:
    def __call__(self, text, return_tensors):
        del text, return_tensors
        return SimpleNamespace(input_ids=torch.arange(12).unsqueeze(0))


class LocalC4LoaderTests(unittest.TestCase):
    def test_c4_calibration_uses_local_train_and_validation_datasets(self):
        train_data = _TextDataset()
        validation_data = _TextDataset()

        with patch(
            "lib.data.load_dataset",
            side_effect=AssertionError("online C4 loading is forbidden"),
        ), patch(
            "lib.data.load_from_disk",
            side_effect=[train_data, validation_data],
            create=True,
        ) as load_from_disk:
            trainloader, validation = get_c4(
                nsamples=1,
                seed=0,
                seqlen=4,
                tokenizer=_Tokenizer(),
            )

        self.assertEqual(len(trainloader), 1)
        self.assertEqual(validation.input_ids.shape, (1, 12))
        self.assertEqual(
            [call.args[0] for call in load_from_disk.call_args_list],
            [
                "/data/DataSet/llm_dataset/c4/c4train",
                "/data/DataSet/llm_dataset/c4/c4validation",
            ],
        )


class LocalWikiTextLoaderTests(unittest.TestCase):
    def test_wikitext_evaluation_uses_local_train_and_test_datasets(self):
        train_data = _TextDataset()
        test_data = _TextDataset()

        with patch(
            "lib.data.load_dataset",
            side_effect=AssertionError("online WikiText loading is forbidden"),
        ), patch(
            "lib.data.load_from_disk",
            side_effect=[train_data, test_data],
        ) as load_from_disk:
            trainloader, testenc = get_wikitext2(
                nsamples=1,
                seed=0,
                seqlen=4,
                tokenizer=_Tokenizer(),
            )

        self.assertEqual(len(trainloader), 1)
        self.assertEqual(testenc.input_ids.shape, (1, 12))
        self.assertEqual(
            [call.args[0] for call in load_from_disk.call_args_list],
            [
                "/data/DataSet/llm_dataset/wikitext2/wikitext2_train",
                "/data/DataSet/llm_dataset/wikitext2/wikitext2_test",
            ],
        )


if __name__ == "__main__":
    unittest.main()
