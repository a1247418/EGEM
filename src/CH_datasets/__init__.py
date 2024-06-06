import sys

from .CH_datasets import utils
try:
	sys.modules["CH_datasets.utils"] = sys.modules["CH_datasets.CH_datasets.utils"]
except Exception as e: print(repr(e))

from .CH_datasets import poisoner
try:
	sys.modules["CH_datasets.poisoner"] = sys.modules["CH_datasets.CH_datasets.poisoner"]
except Exception as e: print(repr(e))

from .CH_datasets import datasets
try:
	sys.modules["CH_datasets.datasets"] = sys.modules["CH_datasets.CH_datasets.datasets"]
except Exception as e: pass

from .CH_datasets import scenario
try:
	sys.modules["CH_datasets.scenario"] = sys.modules["CH_datasets.CH_datasets.scenario"]
except Exception as e: pass

from .CH_datasets import scenario_examples
try:
	sys.modules["CH_datasets.scenario_examples"] = sys.modules["CH_datasets.CH_datasets.scenario_examples"]
except Exception as e: pass
