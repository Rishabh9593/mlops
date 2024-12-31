from sklearn.linear_model import LinearRegression

def test_model():
    model = LinearRegression()
    assert model is not None
