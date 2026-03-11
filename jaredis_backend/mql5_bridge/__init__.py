"""
MQL5 Bridge: Integration layer between Python backend and MetaTrader 5
"""

from .mql5_connector import MQL5Connector
from .signal_communicator import SignalCommunicator

__all__ = ["MQL5Connector", "SignalCommunicator"]
