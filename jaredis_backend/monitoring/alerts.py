"""
Production Monitoring & Alerting System
Real-time system health, performance metrics, and alerts
"""

import logging
import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: str
    condition: Callable[[float], bool]
    level: AlertLevel
    message_template: str
    cooldown_seconds: int = 300  # Don't repeat alerts too frequently


class MetricsCollector:
    """Collect and expose system and trading metrics"""
    
    def __init__(self):
        self.metrics = {}
        
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus()
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics"""
        # Trading metrics
        self.trades_executed = Counter(
            'trades_executed_total',
            'Total trades executed',
            ['symbol', 'action']
        )
        
        self.trade_pnl = Gauge(
            'trade_pnl',
            'Trade profit/loss',
            ['symbol']
        )
        
        self.win_rate = Gauge(
            'trading_win_rate',
            'Win rate percentage'
        )
        
        self.sharpe_ratio = Gauge(
            'sharpe_ratio',
            'Sharpe ratio'
        )
        
        # Account metrics
        self.account_balance = Gauge(
            'account_balance',
            'Account balance'
        )
        
        self.account_equity = Gauge(
            'account_equity',
            'Account equity'
        )
        
        self.margin_level = Gauge(
            'margin_level',
            'Margin level percentage'
        )
        
        self.drawdown = Gauge(
            'account_drawdown',
            'Account drawdown percentage'
        )
        
        # System metrics
        self.connection_active = Gauge(
            'mt5_connection_active',
            'MT5 connection status (0=inactive, 1=active)'
        )
        
        self.data_latency_ms = Histogram(
            'data_latency_milliseconds',
            'Data latency in milliseconds'
        )
        
        self.execution_latency_ms = Histogram(
            'execution_latency_milliseconds',
            'Order execution latency in milliseconds'
        )
        
        # ML metrics
        self.model_prediction_time = Histogram(
            'model_prediction_time_ms',
            'Model prediction time in milliseconds'
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Current model accuracy/F1 score'
        )
    
    def record_trade(self, symbol: str, action: str, pnl: float):
        """Record executed trade"""
        if PROMETHEUS_AVAILABLE:
            self.trades_executed.labels(symbol=symbol, action=action).inc()
            self.trade_pnl.labels(symbol=symbol).set(pnl)
    
    def update_account_metrics(self, metrics: Dict):
        """Update account metrics"""
        if PROMETHEUS_AVAILABLE:
            self.account_balance.set(metrics.get('balance', 0))
            self.account_equity.set(metrics.get('equity', 0))
            self.margin_level.set(metrics.get('margin_level', 0))
            self.drawdown.set(metrics.get('drawdown', 0))
    
    def record_latency(self, latency_ms: float, metric_type: str = 'data'):
        """Record latency measurement"""
        if PROMETHEUS_AVAILABLE:
            if metric_type == 'data':
                self.data_latency_ms.observe(latency_ms)
            elif metric_type == 'execution':
                self.execution_latency_ms.observe(latency_ms)
    
    def record_prediction(self, latency_ms: float, accuracy: float):
        """Record model prediction"""
        if PROMETHEUS_AVAILABLE:
            self.model_prediction_time.observe(latency_ms)
            self.model_accuracy.set(accuracy)


class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self):
        self.alert_rules: List[AlertRule] = []
        self.alert_handlers: List[Callable] = []
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_history: List[Dict] = []
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.alert_rules.append(rule)
        logger.info(f"Alert rule added: {rule.name}")
    
    def add_handler(self, handler: Callable):
        """Add alert handler (e.g., Telegram, Email)"""
        self.alert_handlers.append(handler)
    
    def evaluate_rules(self, metrics: Dict):
        """Evaluate all alert rules"""
        for rule in self.alert_rules:
            if rule.metric not in metrics:
                continue
            
            value = metrics[rule.metric]
            
            if rule.condition(value):
                self._trigger_alert(rule, value)
    
    async def _trigger_alert(self, rule: AlertRule, value: float):
        """Trigger alert if cooldown passed"""
        now = datetime.now()
        last_time = self.last_alert_time.get(rule.name)
        
        # Check cooldown
        if last_time and (now - last_time).seconds < rule.cooldown_seconds:
            return
        
        # Format message
        message = rule.message_template.format(value=value)
        
        # Record alert
        alert = {
            'rule_name': rule.name,
            'level': rule.level.value,
            'message': message,
            'value': value,
            'timestamp': now.isoformat()
        }
        
        self.alert_history.append(alert)
        self.last_alert_time[rule.name] = now
        
        logger.warning(f"[{rule.level.value}] {message}")
        
        # Send to handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")


class TelegramAlertHandler:
    """Send alerts to Telegram"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.webhook_url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    async def __call__(self, alert: Dict):
        """Send alert to Telegram"""
        try:
            import aiohttp
            
            emoji_map = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'error': '🚨',
                'critical': '🛑'
            }
            
            emoji = emoji_map.get(alert['level'], 'ℹ️')
            timestamp = alert['timestamp']
            
            message = f"""
{emoji} *[{alert['level'].upper()}]* Trading Alert

*Name:* {alert['rule_name']}
*Message:* {alert['message']}
*Time:* {timestamp}
            """
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                }
                async with session.post(self.webhook_url, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to send Telegram alert: {resp.status}")
                        
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")


class SMSAlertHandler:
    """Send alerts via SMS using Twilio"""

    def __init__(self, account_sid: str, auth_token: str, from_number: str, to_numbers: List[str]):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_numbers = to_numbers

        try:
            from twilio.rest import Client
            self.twilio_client = Client(account_sid, auth_token)
            self.twilio_available = True
        except ImportError:
            logger.warning("Twilio not installed. SMS alerts disabled.")
            self.twilio_available = False

    async def __call__(self, alert: Dict):
        """Send alert via SMS"""
        if not self.twilio_available:
            logger.warning("Twilio not available for SMS alerts")
            return

        try:
            level_emoji = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'error': '🚨',
                'critical': '🛑'
            }

            emoji = level_emoji.get(alert['level'], 'ℹ️')

            # SMS messages need to be concise
            message = f"{emoji}[{alert['level'].upper()}] {alert['rule_name']}: {alert['message']}"

            # Truncate if too long for SMS
            if len(message) > 160:
                message = message[:157] + "..."

            for to_number in self.to_numbers:
                try:
                    self.twilio_client.messages.create(
                        body=message,
                        from_=self.from_number,
                        to=to_number
                    )
                    logger.info(f"SMS alert sent to {to_number}")
                except Exception as e:
                    logger.error(f"Failed to send SMS to {to_number}: {e}")

        except Exception as e:
            logger.error(f"SMS alert failed: {e}")


class MultiChannelAlertHandler:
    """Handle alerts across multiple channels (Telegram, SMS, Email)"""

    def __init__(self, telegram_handler=None, sms_handler=None, email_handler=None):
        self.handlers = []
        if telegram_handler:
            self.handlers.append(telegram_handler)
        if sms_handler:
            self.handlers.append(sms_handler)
        if email_handler:
            self.handlers.append(email_handler)

    async def __call__(self, alert: Dict):
        """Send alert to all configured channels"""
        tasks = []
        for handler in self.handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(alert))
            else:
                # Run sync handlers in thread pool
                import concurrent.futures
                loop = asyncio.get_event_loop()
                tasks.append(loop.run_in_executor(None, handler, alert))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class EmailAlertHandler:
    """Send alerts via email"""

    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str,
                 from_email: str, to_emails: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            self.email_available = True
        except ImportError:
            logger.warning("Email libraries not available. Email alerts disabled.")
            self.email_available = False

    async def __call__(self, alert: Dict):
        """Send alert via email"""
        if not self.email_available:
            return

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"Trading Alert: {alert['rule_name']} [{alert['level'].upper()}]"

            # HTML body
            html_body = f"""
            <html>
            <body>
                <h2 style="color: {'red' if alert['level'] == 'critical' else 'orange' if alert['level'] == 'error' else 'blue'};">
                    {alert['level'].upper()} Alert: {alert['rule_name']}
                </h2>
                <p><strong>Message:</strong> {alert['message']}</p>
                <p><strong>Value:</strong> {alert['value']}</p>
                <p><strong>Time:</strong> {alert['timestamp']}</p>
                <hr>
                <p><small>This is an automated alert from Jaredis Smart Trading Bot</small></p>
            </body>
            </html>
            """

            msg.attach(MIMEText(html_body, 'html'))

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.from_email, self.to_emails, text)
            server.quit()

            logger.info(f"Email alert sent to {len(self.to_emails)} recipients")

        except Exception as e:
            logger.error(f"Email alert failed: {e}")


class AlertEscalationManager:
    """Escalate alerts based on severity and persistence"""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.escalation_rules = {
            AlertLevel.WARNING: {
                'threshold': 3,  # 3 warnings trigger escalation
                'escalate_to': AlertLevel.ERROR,
                'time_window': 3600  # 1 hour window
            },
            AlertLevel.ERROR: {
                'threshold': 2,  # 2 errors trigger escalation
                'escalate_to': AlertLevel.CRITICAL,
                'time_window': 1800  # 30 minutes window
            }
        }
        self.alert_counts: Dict[str, List[datetime]] = {}

    def check_escalation(self, alert: Dict):
        """Check if alert should be escalated"""
        rule_name = alert['rule_name']
        level = AlertLevel(alert['level'])
        now = datetime.now()

        if level not in self.escalation_rules:
            return

        rule = self.escalation_rules[level]

        # Track alert timestamps
        if rule_name not in self.alert_counts:
            self.alert_counts[rule_name] = []

        # Remove old alerts outside time window
        cutoff = now - timedelta(seconds=rule['time_window'])
        self.alert_counts[rule_name] = [
            ts for ts in self.alert_counts[rule_name] if ts > cutoff
        ]

        # Add current alert
        self.alert_counts[rule_name].append(now)

        # Check if threshold exceeded
        if len(self.alert_counts[rule_name]) >= rule['threshold']:
            escalated_alert = alert.copy()
            escalated_alert['level'] = rule['escalate_to'].value
            escalated_alert['rule_name'] = f"ESCALATED: {rule_name}"
            escalated_alert['message'] = f"ESCALATED: {alert['message']} (occurred {len(self.alert_counts[rule_name])} times)"

            # Trigger escalated alert
            asyncio.create_task(self._trigger_escalated_alert(escalated_alert))

    async def _trigger_escalated_alert(self, alert: Dict):
        """Trigger escalated alert"""
        logger.critical(f"ESCALATED ALERT: {alert['message']}")

        # Send to all handlers
        for handler in self.alert_manager.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Escalated alert handler failed: {e}")


class AlertDashboard:
    """Web dashboard for viewing alerts and metrics"""

    def __init__(self, alert_manager: AlertManager, metrics_collector: MetricsCollector):
        self.alert_manager = alert_manager
        self.metrics_collector = metrics_collector
        self.app = None

    def create_dashboard(self):
        """Create Flask dashboard app"""
        try:
            from flask import Flask, jsonify, render_template_string

            self.app = Flask(__name__)

            @self.app.route('/')
            def dashboard():
                return render_template_string(self._get_dashboard_html())

            @self.app.route('/api/alerts')
            def get_alerts():
                return jsonify(self.alert_manager.alert_history[-50:])  # Last 50 alerts

            @self.app.route('/api/metrics')
            def get_metrics():
                if PROMETHEUS_AVAILABLE:
                    return generate_latest()
                return jsonify({})

            return self.app

        except ImportError:
            logger.warning("Flask not available. Dashboard disabled.")
            return None

    def _get_dashboard_html(self) -> str:
        """Get HTML template for dashboard"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Jaredis Trading Alerts Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .alert { padding: 10px; margin: 5px 0; border-radius: 5px; }
                .info { background-color: #e3f2fd; }
                .warning { background-color: #fff3e0; }
                .error { background-color: #ffebee; }
                .critical { background-color: #fce4ec; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Jaredis Smart Trading Bot - Alert Dashboard</h1>

            <h2>Recent Alerts</h2>
            <div id="alerts"></div>

            <h2>Key Metrics</h2>
            <div id="metrics"></div>

            <script>
                function updateAlerts() {
                    fetch('/api/alerts')
                        .then(r => r.json())
                        .then(alerts => {
                            const container = document.getElementById('alerts');
                            container.innerHTML = alerts.map(alert => `
                                <div class="alert ${alert.level}">
                                    <strong>${alert.level.toUpperCase()}</strong> ${alert.rule_name}<br>
                                    ${alert.message}<br>
                                    <small>${alert.timestamp}</small>
                                </div>
                            `).join('');
                        });
                }

                function updateMetrics() {
                    fetch('/api/metrics')
                        .then(r => r.text())
                        .then(metrics => {
                            const container = document.getElementById('metrics');
                            // Parse Prometheus format and display key metrics
                            container.innerHTML = '<p>Metrics loading...</p>';
                        });
                }

                setInterval(updateAlerts, 5000);
                setInterval(updateMetrics, 10000);
                updateAlerts();
                updateMetrics();
            </script>
        </body>
        </html>
        """

    def run_dashboard(self, host='0.0.0.0', port=8080):
        """Run the dashboard server"""
        if self.app:
            logger.info(f"Starting alert dashboard on {host}:{port}")
            self.app.run(host=host, port=port, debug=False)
        else:
            logger.error("Dashboard not available - Flask not installed")
    """Monitor trading performance metrics"""
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_history: List[float] = []
        self.daily_metrics: Dict = {}
    
    def add_trade(self, trade: Dict):
        """Record trade"""
        self.trades.append(trade)
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {}
        
        # Win rate
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        win_rate = winning_trades / len(self.trades) if self.trades else 0
        
        # Profit factor
        total_profit = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
        total_loss = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Return metrics
        pnls = [t.get('pnl', 0) for t in self.trades]
        total_pnl = sum(pnls)
        
        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            import numpy as np
            returns = np.array(pnls)
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'avg_trade': total_pnl / len(self.trades) if self.trades else 0
        }
    
    def get_daily_summary(self) -> Dict:
        """Get today's trading summary"""
        from datetime import date
        today = date.today()
        
        today_trades = [t for t in self.trades 
                       if t.get('time', datetime.now()).date() == today]
        
        daily_pnl = sum(t.get('pnl', 0) for t in today_trades)
        
        return {
            'date': today.isoformat(),
            'trades_today': len(today_trades),
            'daily_pnl': daily_pnl,
            'trades': today_trades
        }


class SystemHealthMonitor:
    """Monitor system health and availability"""
    
    def __init__(self):
        self.connection_active = False
        self.last_data_time = None
        self.last_execution_time = None
        self.error_count = 0
        self.data_latencies = []
        self.execution_latencies = []
    
    def check_health(self) -> Dict[str, bool]:
        """Check overall system health"""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        health = {
            'mt5_connected': self.connection_active,
            'data_fresh': self.last_data_time and (now - self.last_data_time) < timedelta(seconds=60),
            'execution_working': self.last_execution_time and (now - self.last_execution_time) < timedelta(minutes=5),
            'error_rate_low': self.error_count < 10,
            'average_latency_low': self._avg_latency() < 1000  # ms
        }
        
        return health
    
    def _avg_latency(self) -> float:
        """Calculate average latency"""
        all_latencies = self.data_latencies + self.execution_latencies
        if all_latencies:
            return sum(all_latencies) / len(all_latencies)
        return 0


# Configuration for common alert rules
DEFAULT_ALERT_RULES = [
    AlertRule(
        name="High Drawdown",
        metric="drawdown",
        condition=lambda x: x > 0.08,  # 8%
        level=AlertLevel.WARNING,
        message_template="Drawdown high: {value:.2%}"
    ),
    AlertRule(
        name="Critical Drawdown",
        metric="drawdown",
        condition=lambda x: x > 0.10,  # 10%
        level=AlertLevel.CRITICAL,
        message_template="CRITICAL drawdown: {value:.2%}"
    ),
    AlertRule(
        name="Low Win Rate",
        metric="win_rate",
        condition=lambda x: x < 0.45,
        level=AlertLevel.WARNING,
        message_template="Win rate below 45%: {value:.2%}"
    ),
    AlertRule(
        name="High Execution Latency",
        metric="execution_latency_ms",
        condition=lambda x: x > 500,  # 500ms
        level=AlertLevel.WARNING,
        message_template="Execution latency high: {value:.0f}ms"
    ),
    AlertRule(
        name="MT5 Connection Lost",
        metric="mt5_connected",
        condition=lambda x: x == 0,
        level=AlertLevel.CRITICAL,
        message_template="MT5 connection lost!"
    ),
]
