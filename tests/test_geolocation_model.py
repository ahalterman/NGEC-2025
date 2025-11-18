# Smoke test for geolocation model

from ngec import GeolocationModel
import pytest

def test_geolocation_model():
    geolocation_model = GeolocationModel(geo_model=None, geo_path=None)

    assert geolocation_model is not None