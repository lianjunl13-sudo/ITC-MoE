
import json
import re

from datasets import Dataset, DatasetDict

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import TEXT_POSTPROCESSORS


class ScreenMathDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        with open(path, "r", encoding="utf-8") as input_file:
            rows = [json.loads(line) for line in input_file if line.strip()]
        test_dataset = Dataset.from_list(rows)
        
        return DatasetDict(train=test_dataset.select([]), test=test_dataset)


@TEXT_POSTPROCESSORS.register_module(
    name="second_batch_last_number", force=True
)
def last_number(text: str) -> str:
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else "NULL"


class NumericEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        details = []
        correct_count = 0
        for prediction, reference in zip(predictions, references):
            try:
                correct = abs(float(prediction) - float(reference)) < 1e-6
            except (TypeError, ValueError):
                correct = False
            correct_count += int(correct)
            details.append(
                {"pred": prediction, "answer": reference, "correct": correct}
            )
        return {
            "accuracy": 100.0 * correct_count / len(references),
            "details": details,
        }
