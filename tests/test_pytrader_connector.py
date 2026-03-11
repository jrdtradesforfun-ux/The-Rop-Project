"""
Test Pytrader Connector integration
"""

import pytest

from jaredis_backend.mql5_bridge import PytraderConnector


def test_pytrader_connector_import():
    """Verify that the PytraderConnector class can be imported and instantiated"""
    connector = PytraderConnector(host="localhost", port=5000)
    assert not connector.connected
    # check basic attributes
    assert hasattr(connector, "connect")
    assert hasattr(connector, "get_instruments")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
