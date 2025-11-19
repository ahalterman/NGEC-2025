
import pytest
import spacy

from ngec import AttributeModel

@pytest.fixture(scope='session', autouse=True)
def nlp():
    return spacy.load("en_core_web_trf")

@pytest.fixture(scope='session', autouse=True)
def am():
    return AttributeModel(silent=True, gpu=False, backend="transformers")