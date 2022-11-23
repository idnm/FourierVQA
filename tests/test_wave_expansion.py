import wave_expansion as we


def test_trivial():
    assert we.PauliString.trivial(2) == we.PauliString([0, 0], [0, 0])


def test_true():
    assert True