import pickle
import os


def test_model_exists():
    assert os.path.exists('models/model.pkl')


def test_model_loading():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    assert model is not None
