#!/usr/bin/env python3
"""Quick Setup Guide for Production Trading Bot"""

print("🚀 Jaredis Smart Trading Bot - Production Setup Guide")
print("=" * 60)

print("\n📋 STEP 1: Install Dependencies")
print("pip install -r requirements.txt")

print("\n📋 STEP 2: Configure Cloud Storage (Optional)")
print("Edit production_trading_bot.py and add your credentials:")
print("- AWS S3: bucket name, access key, secret key")
print("- Google Cloud: service account JSON file")
print("- Azure: connection string")

print("\n📋 STEP 3: Configure Alerting Channels")
print("Edit production_trading_bot.py with your tokens:")
print("- Telegram: Bot token and chat ID")
print("- Twilio: Account SID, auth token, phone numbers")
print("- Email: SMTP server, credentials")

print("\n📋 STEP 4: Test Individual Components")

print("\n🔧 Test Backup System:")
print("python test_backup.py")

print("\n🔧 Test GPU Acceleration:")
print("python -c \"from jaredis_backend.ml_models.gpu_utils import enable_gpu_acceleration; enable_gpu_acceleration()\"")

print("\n📋 STEP 5: Run Production Bot")
print("python production_trading_bot.py --gpu --backup --alerts")

print("\n📋 STEP 6: Monitor and Maintain")
print("- Check logs/ directory for system logs")
print("- Backups stored in backups/ directory")
print("- Web dashboard: http://localhost:8080 (if enabled)")

print("\n✅ Your bot is now production-ready!")
print("💡 Pro tip: Start with --gpu only, then add --backup and --alerts")