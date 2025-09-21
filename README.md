# 🤖 AI Trading Education System

An educational platform that demonstrates how AI agents analyze stocks and provide trading insights. Built for learning purposes only.

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd agentic-trading-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv trading_env
   source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run trading_education_website.py
   ```

### Streamlit Cloud Deployment

1. **Push to GitHub**
   - Make sure your repository contains:
     - `trading_education_website.py`
     - `free_ai_trading_system.py`
     - `requirements.txt`

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file: `trading_education_website.py`
   - Deploy!

## 📦 Dependencies

The application requires these main packages:
- `streamlit>=1.28.0` - Web app framework
- `plotly>=5.17.0` - Interactive charts
- `yfinance>=0.2.28` - Market data
- `pandas>=2.1.0` - Data manipulation
- `numpy>=1.25.0` - Numerical computing
- `requests>=2.31.0` - HTTP requests
- `python-dotenv>=1.0.0` - Environment variables

## 🔧 Troubleshooting

### Import Errors
If you get import errors like `ModuleNotFoundError: No module named 'plotly'`:

1. **Check if packages are installed:**
   ```bash
   pip list | grep plotly
   pip list | grep yfinance
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **For Streamlit Cloud:**
   - Make sure `requirements.txt` is in the root directory
   - Check the deployment logs for any installation errors
   - Try redeploying the app

### Common Issues

1. **Plotly not found**: Install with `pip install plotly>=5.17.0`
2. **yfinance not found**: Install with `pip install yfinance>=0.2.28`
3. **Charts not displaying**: Check that both plotly and yfinance are properly installed

## 🎓 Educational Features

- **Multi-Agent AI Analysis**: See how different AI agents analyze the same stock
- **Technical Indicators**: Learn about RSI, Moving Averages, and more
- **Interactive Charts**: Explore price data with candlestick charts
- **Risk Management**: Understand trading psychology and risk concepts

## ⚠️ Disclaimer

**This platform is for educational purposes only and does not constitute financial advice.**

- Learn how AI agents analyze market data
- Understand technical indicators and patterns
- See how multiple AI perspectives combine
- Practice with paper trading concepts

Always do your own research and consult financial professionals before making investment decisions.

## 🤝 Support

If this educational tool helps you learn about AI and trading, consider supporting the project:

☕ [Buy me a coffee](https://www.buymeacoffee.com/yourusername)

## 📄 License

This project is open source and available under the MIT License.
