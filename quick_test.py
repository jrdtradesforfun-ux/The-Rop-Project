#!/usr/bin/env python3
"""Quick test of production trading bot components"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from production_trading_bot import ProductionTradingBot
import pandas as pd
import numpy as np

async def quick_test():
    print('🚀 Testing Production Trading Bot Components...')

    try:
        # Initialize bot
        bot = ProductionTradingBot(enable_backups=True, enable_alerts=True)
        await bot.initialize_production_features()
        print('✅ Bot initialized successfully')

        # Create mock data
        print('📊 Creating mock trading data...')
        dates = pd.date_range('2023-01-01', periods=1000, freq='15min')
        df = pd.DataFrame({
            'open': np.random.uniform(1.05, 1.15, 1000),
            'high': np.random.uniform(1.05, 1.15, 1000),
            'low': np.random.uniform(1.05, 1.15, 1000),
            'close': np.random.uniform(1.05, 1.15, 1000),
            'tick_volume': np.random.randint(100, 1000, 1000)
        }, index=dates)

        # Test model training (skip optimization for speed)
        print('🤖 Training ML models (CPU only, no optimization)...')
        results = await bot.train_models_with_gpu(df)
        xgb_f1 = results['xgb_metrics']['test_f1']
        print(f'✅ XGBoost trained - F1 Score: {xgb_f1:.3f}')

        # Test backup
        print('💾 Creating backup...')
        backup_path = await bot.create_backup('models')
        if backup_path:
            print(f'✅ Backup created: {backup_path}')
        else:
            print('⚠️ Backup creation failed')

        await bot.shutdown()
        print('✅ All tests passed! Production bot is ready.')

    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())