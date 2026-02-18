
from ngec.actors.actor_resolution import CountryDetector


def test_country_detector():
    cd = CountryDetector()
    res = cd.search_nat("There were also 5 Americans in the village.")
    assert res == ('USA', 'There were also 5 in the village.')


def test_country_detector_no_country():
    cd = CountryDetector()
    res = cd.search_nat("This text has no country mentioned.")
    assert res == (None, 'This text has no country mentioned.')