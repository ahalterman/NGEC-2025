from NGEC import ActorResolver
from NGEC.actor_resolution import WikiMatcher
import pytest
import spacy
from NGEC import AttributeModel

@pytest.fixture(scope='session', autouse=True)
def ag():
    return ActorResolver(base_path="./assets/")

@pytest.fixture(scope='session', autouse=True)
def matcher():
    return WikiMatcher()

@pytest.fixture(scope='session', autouse=True)
def nlp():
    return spacy.load("en_core_web_trf")

@pytest.fixture(scope='session', autouse=True)
def am():
    return AttributeModel()

