import pytest
from utils import generateTestdataPython

mergedTestData = generateTestdataPython()


@pytest.mark.parametrize("lib_pred,actual,model", mergedTestData)
def test_predict(lib_pred, actual, model):
    boolResult = abs(lib_pred - actual) <= 0.001
    if not boolResult:
        print("Error occurred for --> ", model)

    assert boolResult
