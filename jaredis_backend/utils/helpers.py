"""
Helper Functions: Common utility functions
"""

from typing import Dict, List


def format_signal(signal: Dict) -> str:
    """Format signal for display/logging"""
    return (
        f"Signal: {signal.get('direction', 'UNKNOWN')} "
        f"{signal.get('symbol', 'N/A')} @ {signal.get('entry_price', 0):.2f} "
        f"SL:{signal.get('stop_loss', 0):.2f} TP:{signal.get('take_profit', 0):.2f}"
    )


def calculate_metrics(trades: List[Dict]) -> Dict:
    """
    Calculate trading performance metrics
    
    Args:
        trades: List of completed trades
        
    Returns:
        Performance metrics dictionary
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "total_pnl": 0
        }

    pnls = [t.get("pnl", 0) for t in trades if t.get("status") == "closed"]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    total_pnl = sum(pnls)

    profit_factor = avg_win / abs(avg_loss) if avg_loss != 0 else 0

    return {
        "total_trades": len(pnls),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl
    }


def validate_required_fields(data: Dict, required: List[str]) -> bool:
    """Check if all required fields are present in data"""
    return all(field in data for field in required)
