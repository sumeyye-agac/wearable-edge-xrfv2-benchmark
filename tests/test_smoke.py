from xrfv2_edge_tal import __version__


def test_version_defined() -> None:
    assert isinstance(__version__, str)
    assert len(__version__) > 0
