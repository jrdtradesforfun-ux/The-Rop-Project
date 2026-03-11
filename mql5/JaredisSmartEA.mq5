//+------------------------------------------------------------------+
//| Jaredis Smart Trading EA for MetaTrader 5                        |
//| Communicates with Python Trading Bot via Socket                  |
//+------------------------------------------------------------------+

#property copyright "Jaredis Smart"
#property link "https://github.com/jrdtradesforfun-ux/The-Rop-Project"
#property version "1.0"
#property strict

#include <Trade/Trade.mqh>
#include <Trade/OrderInfo.mqh>

//+------------------------------------------------------------------+
//| Socket Communications                                            |
//+------------------------------------------------------------------+

int server_socket = INVALID_HANDLE;
string python_host = "localhost";
int python_port = 5000;

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+

CTrade trade;
COrderInfo order_info;

int magic_number = 12345;
int timeout_connection = 5000;  // 5 seconds
int timeout_socket = 1000;      // 1 second

struct OrderSignal {
    string symbol;
    string direction;       // "long" or "short"
    double entry_price;
    double stop_loss;
    double take_profit;
    double volume;
    string signal_type;
};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

int OnInit() {
    Print("Jaredis Smart EA initializing...");
    
    // Set magic number for trade identification
    trade.SetExpertMagicNumber(magic_number);
    
    // Initialize socket connection
    if (!InitializeSocket()) {
        Print("Failed to initialize socket connection");
        return INIT_FAILED;
    }
    
    Print("Jaredis Smart EA initialized successfully");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+

void OnDeinit(const int reason) {
    Print("Jaredis Smart EA deinitialization");
    
    if (server_socket != INVALID_HANDLE) {
        SocketClose(server_socket);
        server_socket = INVALID_HANDLE;
    }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+

void OnTick() {
    static datetime last_check = 0;
    
    // Check for signals every 5 seconds
    if (TimeCurrent() - last_check < 5) {
        return;
    }
    last_check = TimeCurrent();
    
    // Receive and process signals from Python
    ProcessIncomingSignals();
    
    // Monitor open positions
    MonitorOpenPositions();
}

//+------------------------------------------------------------------+
//| Socket Initialization                                            |
//+------------------------------------------------------------------+

bool InitializeSocket() {
    Print("Initializing socket connection to Python bot...");
    
    server_socket = SocketCreate();
    if (server_socket == INVALID_HANDLE) {
        Print("Failed to create socket");
        return false;
    }
    
    Print("Socket created. Attempting connection to ", python_host, ":", python_port);
    
    if (!SocketConnect(server_socket, python_host, python_port, timeout_connection)) {
        Print("Failed to connect to Python bot at ", python_host, ":", python_port);
        SocketClose(server_socket);
        server_socket = INVALID_HANDLE;
        return false;
    }
    
    Print("Connected to Python bot successfully");
    return true;
}

//+------------------------------------------------------------------+
//| Receive Signals from Python Bot                                  |
//+------------------------------------------------------------------+

void ProcessIncomingSignals() {
    if (server_socket == INVALID_HANDLE) {
        // Try to reconnect
        if (!InitializeSocket()) {
            return;
        }
    }
    
    // Buffer for incoming data
    char buffer[1024];
    int bytes_received = SocketReceive(server_socket, buffer, sizeof(buffer), timeout_socket);
    
    if (bytes_received > 0) {
        string message(buffer);
        
        // Remove null terminators
        StringTrimRight(message);
        
        Print("Received signal from Python: ", message);
        
        // Parse and execute signal
        OrderSignal signal = ParseSignal(message);
        if (signal.symbol != "") {
            ExecuteSignal(signal);
        }
    } else if (bytes_received < 0) {
        Print("Socket error detected, attempting reconnect");
        SocketClose(server_socket);
        server_socket = INVALID_HANDLE;
    }
}

//+------------------------------------------------------------------+
//| Parse Signal from Python                                         |
//+------------------------------------------------------------------+

OrderSignal ParseSignal(const string signal_json) {
    OrderSignal signal;
    signal.symbol = "";
    signal.volume = 0;
    
    // Simple JSON parsing (more robust parsing recommended for production)
    // Example format: {"symbol":"EURUSD","direction":"long","volume":0.1,...}
    
    if (signal_json.Find("symbol") < 0) {
        return signal;
    }
    
    // Extract symbol
    int start = signal_json.Find(":\"") + 2;
    int end = signal_json.Find("\"", start);
    if (start > 1 && end > start) {
        signal.symbol = StringSubstr(signal_json, start, end - start);
    }
    
    // Extract direction (simple parsing)
    if (signal_json.Find("long") > 0) {
        signal.direction = "long";
    } else if (signal_json.Find("short") > 0) {
        signal.direction = "short";
    } else {
        return signal;
    }
    
    // Extract volume (look for "volume":0.1 pattern)
    int vol_pos = signal_json.Find("\"volume\":");
    if (vol_pos > 0) {
        string vol_str = StringSubstr(signal_json, vol_pos + 10, 5);
        signal.volume = StringToDouble(vol_str);
    }
    
    // Extract stop loss
    int sl_pos = signal_json.Find("\"stop_loss\":");
    if (sl_pos > 0) {
        string sl_str = StringSubstr(signal_json, sl_pos + 12, 10);
        signal.stop_loss = StringToDouble(sl_str);
    }
    
    // Extract take profit
    int tp_pos = signal_json.Find("\"take_profit\":");
    if (tp_pos > 0) {
        string tp_str = StringSubstr(signal_json, tp_pos + 14, 10);
        signal.take_profit = StringToDouble(tp_str);
    }
    
    return signal;
}

//+------------------------------------------------------------------+
//| Execute Trading Signal                                           |
//+------------------------------------------------------------------+

void ExecuteSignal(const OrderSignal &signal) {
    if (signal.symbol == "" || signal.volume <= 0) {
        Print("Invalid signal received");
        return;
    }
    
    // Validate symbol
    if (!SymbolSelect(signal.symbol, true)) {
        Print("Symbol not available: ", signal.symbol);
        return;
    }
    
    Print("Executing signal for ", signal.symbol, 
          " Direction: ", signal.direction, 
          " Volume: ", signal.volume);
    
    // Get symbol info
    MqlTick tick;
    if (!SymbolInfoTick(signal.symbol, tick)) {
        Print("Failed to get tick for ", signal.symbol);
        return;
    }
    
    // Determine order type
    ENUM_ORDER_TYPE order_type;
    double entry_price;
    
    if (signal.direction == "long") {
        order_type = ORDER_TYPE_BUY;
        entry_price = tick.ask;
    } else {
        order_type = ORDER_TYPE_SELL;
        entry_price = tick.bid;
    }
    
    // Use SL/TP from signal or defaults
    double sl = signal.stop_loss > 0 ? signal.stop_loss : 0;
    double tp = signal.take_profit > 0 ? signal.take_profit : 0;
    
    // Place order
    if (!trade.PositionOpen(signal.symbol, order_type, signal.volume, entry_price, sl, tp)) {
        Print("Order failed. Error: ", GetLastError());
        Print("Details: ", trade.ResultComment());
    } else {
        Print("Order placed successfully. Ticket: ", trade.ResultOrder());
        
        // Send confirmation back to Python
        SendSignalExecutionStatus(signal.symbol, true, trade.ResultOrder());
    }
}

//+------------------------------------------------------------------+
//| Monitor Open Positions                                           |
//+------------------------------------------------------------------+

void MonitorOpenPositions() {
    // Update profit/loss monitoring
    double total_profit = 0;
    
    for (int i = PositionsTotal() - 1; i >= 0; i--) {
        if (m_position.SelectByIndex(i)) {
            if (m_position.Magic() == magic_number && m_position.PositionType() <= 1) {
                total_profit += m_position.Profit();
            }
        }
    }
    
    // Log total profit (optional)
    if (total_profit != 0) {
        Print("Total profit from EA positions: ", total_profit);
    }
}

//+------------------------------------------------------------------+
//| Send Status Back to Python                                       |
//+------------------------------------------------------------------+

void SendSignalExecutionStatus(const string symbol, bool success, ulong order_ticket) {
    if (server_socket == INVALID_HANDLE) {
        return;
    }
    
    // Format: {"status":"executed","symbol":"EURUSD","ticket":123456}
    string status_msg = "{\"status\":\"" + 
                        (success ? "executed" : "failed") + 
                        "\",\"symbol\":\"" + symbol + 
                        "\",\"ticket\":" + IntegerToString(order_ticket) + "}";
    
    if (!SocketSend(server_socket, status_msg)) {
        Print("Failed to send status to Python");
    }
}

//+------------------------------------------------------------------+
//| Helper: Convert string to double                                 |
//+------------------------------------------------------------------+

double StringToDouble(const string str_value) {
    return StringToDouble(str_value);
}

//+------------------------------------------------------------------+
//| Position Info Helper                                             |
//+------------------------------------------------------------------+

CPositionInfo m_position;
COrderInfo m_order;

//+------------------------------------------------------------------+
//| Socket Functions (High-level wrappers)                          |
//+------------------------------------------------------------------+

int SocketCreate() {
    // Note: In actual MT5 implementation, use built-in socket functions
    // This is a simplified version
    return 1;  // Return valid handle
}

bool SocketConnect(int socket, const string host, int port, int timeout) {
    // Actual implementation uses MT5 API
    // This connects to Python server
    Print("Socket connect attempt: ", host, ":", port);
    return true;
}

int SocketReceive(int socket, char &buffer[], int buffer_size, int timeout) {
    // Receive data from socket
    // Returns bytes received or -1 on error
    return 0;  // Would be replaced with actual socket receive
}

bool SocketSend(int socket, const string message) {
    // Send data through socket
    return true;
}

void SocketClose(int socket) {
    // Close socket connection
    Print("Socket closed");
}

//+------------------------------------------------------------------+
//| END OF EXPERT ADVISOR                                            |
//+------------------------------------------------------------------+
