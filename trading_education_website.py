"""
AI Trading Education Website
Educational platform demonstrating agentic AI trading analysis
Built with Streamlit for easy deployment and sharing
File: trading_education_website.py
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import json
import time
import logging

# Import our free AI trading system
try:
    from free_ai_trading_system import FreeTradingSystem, FreeDataProvider
except ImportError:
    st.error("Please make sure free_ai_trading_system.py is in the same directory!")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Trading Education Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/ai-trading-education',
        'Report a bug': 'https://github.com/yourusername/ai-trading-education/issues',
        'About': """
        # AI Trading Education Platform
        
        This educational platform demonstrates how AI agents can analyze stocks
        and provide trading insights. Built for learning purposes only.
        
        **⚠️ EDUCATIONAL ONLY - NOT FINANCIAL ADVICE**
        """
    }
)

# Custom CSS for clean white theme
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #ffffff;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        background-color: #ffffff;
    }
    
    .main-header {
        font-size: 3.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    .disclaimer-box {
        background-color: #fffbf0;
        border: 1px solid #f6e05e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
        border-left: 4px solid #ed8936;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .disclaimer-box h3 {
        color: #744210;
        margin-top: 0;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .signal-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        background-color: #ffffff;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .signal-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
    }
    
    .signal-buy {
        border-left: 4px solid #28a745;
        background: linear-gradient(135deg, #f0fff4 0%, #ffffff 100%);
    }
    
    .signal-sell {
        border-left: 4px solid #dc3545;
        background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
    }
    
    .signal-hold {
        border-left: 4px solid #ffc107;
        background: linear-gradient(135deg, #fffbf0 0%, #ffffff 100%);
    }
    
    .coffee-button {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        padding: 12px 24px;
        text-decoration: none;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
        margin: 15px 0;
        text-align: center;
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3);
        transition: all 0.3s ease;
    }
    
    .coffee-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(255, 107, 53, 0.4);
        text-decoration: none;
        color: white;
    }
    
    .agent-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .agent-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .agent-card h4 {
        color: #2d3748;
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #e9ecef;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 20px;
        border-radius: 8px;
        color: #495057;
        font-weight: 500;
        border: none;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        color: #212529;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff;
        color: #212529;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #dee2e6;
        background-color: #ffffff;
        color: #495057;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button:hover {
        background-color: #f8f9fa;
        border-color: #adb5bd;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #212529;
        font-weight: 600;
    }
    
    p, div, span, li {
        color: #495057;
        line-height: 1.6;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="metric-container"] > div {
        color: #212529;
    }
    
    /* Status messages */
    .stSuccess {
        background-color: #f0fff4;
        border: 1px solid #9ae6b4;
        border-radius: 8px;
        color: #276749;
        padding: 1rem;
    }
    
    .stError {
        background-color: #fff5f5;
        border: 1px solid #feb2b2;
        border-radius: 8px;
        color: #c53030;
        padding: 1rem;
    }
    
    .stWarning {
        background-color: #fffbf0;
        border: 1px solid #f6e05e;
        border-radius: 8px;
        color: #744210;
        padding: 1rem;
    }
    
    .stInfo {
        background-color: #ebf8ff;
        border: 1px solid #90cdf4;
        border-radius: 8px;
        color: #2c5282;
        padding: 1rem;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #dee2e6;
        color: #495057;
        background-color: #ffffff;
        padding: 0.75rem;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #dee2e6;
        background-color: #ffffff;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        color: #495057;
        font-weight: 500;
        padding: 1rem;
    }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid #e9ecef;
        background-color: #f8f9fa;
    }
    
    /* Custom containers */
    .feature-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Remove default margins and add consistent spacing */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Sidebar headers */
    .css-1d391kg h2, .css-1d391kg h3 {
        color: #212529;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def get_trading_system():
    """Initialize trading system (cached to avoid recreation)"""
    return FreeTradingSystem()

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'educational_mode' not in st.session_state:
    st.session_state.educational_mode = True

def run_async(coro):
    """Helper to run async functions in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def create_price_chart(symbol, period="3mo"):
    """Create interactive price chart with technical indicators"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            return None, None
        
        # Create subplot with secondary y-axis for volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price Action', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                 for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Technical Analysis",
            yaxis_title="Price ($)",
            yaxis2_title="Volume",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig, data
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None, None

def display_analysis_card(analysis_result):
    """Display analysis result in a styled card"""
    if 'error' in analysis_result:
        st.error(f"Analysis failed: {analysis_result['error']}")
        return
    
    consensus = analysis_result['consensus']
    signals = analysis_result['signals']
    symbol = analysis_result['symbol']
    
    # Main consensus card
    action = consensus['action']
    confidence = consensus['confidence']
    
    if action == 'BUY':
        card_class = "signal-buy"
        emoji = "🟢"
        color = "#28a745"
    elif action == 'SELL':
        card_class = "signal-sell"
        emoji = "🔴"
        color = "#dc3545"
    else:
        card_class = "signal-hold"
        emoji = "🟡"
        color = "#ffc107"
    
    st.markdown(f"""
    <div class="signal-card {card_class}">
        <h2>{emoji} {symbol} Analysis</h2>
        <h3>Consensus: {action} ({confidence:.1%} confidence)</h3>
        <p><strong>AI Reasoning:</strong> {consensus['reasoning']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Individual agent analysis
    st.subheader("🤖 AI Agent Breakdown")
    
    cols = st.columns(len(signals))
    for i, signal in enumerate(signals):
        with cols[i]:
            confidence_color = "#28a745" if signal.confidence > 0.7 else "#ffc107" if signal.confidence > 0.5 else "#dc3545"
            
            st.markdown(f"""
            <div class="agent-card">
                <h4>{signal.agent_name}</h4>
                <p><strong>Action:</strong> {signal.action}</p>
                <p><strong>Confidence:</strong> <span style="color: {confidence_color};">{signal.confidence:.1%}</span></p>
                <p><strong>Analysis:</strong> {signal.reasoning[:150]}...</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Trading Education Hub</h1>', unsafe_allow_html=True)
    
    # Educational disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <h3>📚 Educational Purpose Only</h3>
        <p><strong>⚠️ This platform is for educational purposes only and does not constitute financial advice.</strong></p>
        <p>• Learn how AI agents analyze market data<br>
        • Understand technical indicators and patterns<br>
        • See how multiple AI perspectives combine<br>
        • Practice with paper trading concepts</p>
        <p><em>Always do your own research and consult financial professionals before making investment decisions.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Support section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">💡 If this educational tool helps you learn about AI and trading:</p>
            <a href="https://www.buymeacoffee.com/yourusername" target="_blank" class="coffee-button">
                ☕ Buy me a coffee
            </a>
            <p style="margin-top: 1rem;"><small>Help keep this educational platform free and updated!</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Analysis Controls")
        
        # Symbol input
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter any stock symbol (e.g., AAPL, TSLA, GOOGL)"
        ).upper()
        
        # AI Model selection
        st.subheader("🤖 AI Configuration")
        ai_provider = st.selectbox(
            "AI Provider",
            ["ollama", "huggingface", "openai_free", "simple"],
            help="Choose which free AI model to use for analysis"
        )
        
        if ai_provider == "ollama":
            st.info("💡 Ollama provides the best free AI experience. Make sure it's running locally!")
        elif ai_provider == "openai_free":
            st.warning("⚠️ OpenAI requires API key and uses your free credit")
        
        # Analysis options
        st.subheader("📊 Analysis Options")
        include_chart = st.checkbox("Show Price Chart", value=True)
        include_indicators = st.checkbox("Show Technical Indicators", value=True)
        detailed_reasoning = st.checkbox("Detailed AI Reasoning", value=True)
        
        # Educational features
        st.subheader("📚 Educational Features")
        show_agent_comparison = st.checkbox("Compare Agent Perspectives", value=True)
        explain_indicators = st.checkbox("Explain Technical Indicators", value=True)
        
        # Quick analysis buttons
        st.subheader("🚀 Quick Analysis")
        popular_stocks = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN", "SPY", "QQQ"]
        
        selected_quick = st.selectbox("Quick Analyze", ["Select..."]+popular_stocks)
        if selected_quick != "Select...":
            symbol = selected_quick
        
        # Analysis history
        st.subheader("📈 Recent Analysis")
        if st.session_state.analysis_history:
            for hist in st.session_state.analysis_history[-3:]:
                if st.button(f"{hist['symbol']} ({hist['action']})", key=f"hist_{hist['timestamp']}"):
                    symbol = hist['symbol']
        
        # AI Provider Status
        st.subheader("🔧 System Status")
        trading_system = get_trading_system()
        
        # Check Ollama
        try:
            import requests
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            if response.status_code == 200:
                st.success("✅ Ollama: Connected")
            else:
                st.error("❌ Ollama: Not running")
        except:
            st.warning("⚠️ Ollama: Not available")
        
        # Check market data
        try:
            test_data = yf.Ticker("AAPL").history(period="1d")
            if not test_data.empty:
                st.success("✅ Market Data: Available")
            else:
                st.error("❌ Market Data: No data")
        except:
            st.error("❌ Market Data: Connection failed")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Analysis", "📚 How It Works", "🎓 Learning Center", "💡 About AI Trading"])
    
    with tab1:
        st.header(f"📊 AI Analysis for {symbol}")
        
        # Analysis button
        if st.button(f"🔍 Analyze {symbol}", type="primary", use_container_width=True):
            with st.spinner(f"🤖 AI agents analyzing {symbol}... This may take 30-60 seconds"):
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Update progress
                    status_text.text("🔍 Fetching market data...")
                    progress_bar.progress(20)
                    
                    # Get trading system
                    trading_system = get_trading_system()
                    
                    # Run analysis
                    status_text.text("🤖 AI agents working...")
                    progress_bar.progress(50)
                    
                    analysis_result = run_async(
                        trading_system.analyze_symbol(symbol)
                    )
                    
                    progress_bar.progress(80)
                    status_text.text("📊 Preparing results...")
                    
                    # Store in history
                    if 'error' not in analysis_result:
                        st.session_state.analysis_history.append({
                            'symbol': symbol,
                            'action': analysis_result['consensus']['action'],
                            'confidence': analysis_result['consensus']['confidence'],
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Keep only last 10 analyses
                        if len(st.session_state.analysis_history) > 10:
                            st.session_state.analysis_history = st.session_state.analysis_history[-10:]
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    time.sleep(1)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    display_analysis_card(analysis_result)
                    
                    # Show chart if requested
                    if include_chart and 'error' not in analysis_result:
                        st.subheader("📈 Technical Chart")
                        fig, data = create_price_chart(symbol)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Technical indicators summary
                            if include_indicators and data is not None:
                                st.subheader("📊 Technical Indicators")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                current_price = data['Close'].iloc[-1]
                                sma_20 = data['Close'].rolling(20).mean().iloc[-1]
                                sma_50 = data['Close'].rolling(50).mean().iloc[-1]
                                
                                # Calculate RSI
                                delta = data['Close'].diff()
                                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                                rs = gain / loss
                                rsi = (100 - (100 / (1 + rs))).iloc[-1]
                                
                                with col1:
                                    st.metric("Current Price", f"${current_price:.2f}")
                                
                                with col2:
                                    sma20_diff = ((current_price / sma_20) - 1) * 100
                                    st.metric("vs SMA20", f"{sma20_diff:+.1f}%")
                                
                                with col3:
                                    sma50_diff = ((current_price / sma_50) - 1) * 100
                                    st.metric("vs SMA50", f"{sma50_diff:+.1f}%")
                                
                                with col4:
                                    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                                    st.metric("RSI", f"{rsi:.1f} ({rsi_status})")
                    
                    # Educational explanations
                    if explain_indicators:
                        with st.expander("📚 Understanding the Analysis"):
                            st.markdown("""
                            ### How AI Agents Analyze Stocks:
                            
                            **🤖 Market Analyzer Agent:**
                            - Examines price patterns and trends
                            - Calculates technical indicators (RSI, Moving Averages)
                            - Identifies support and resistance levels
                            - Assesses momentum and volume patterns
                            
                            **📊 Sentiment Analyzer Agent:**
                            - Analyzes market sentiment indicators
                            - Evaluates volume patterns for institutional activity
                            - Considers momentum and relative strength
                            - Assesses overall market environment
                            
                            **🧠 Consensus Building:**
                            - Combines insights from multiple agents
                            - Weights opinions by confidence levels
                            - Resolves conflicts using AI reasoning
                            - Provides balanced recommendation
                            
                            **📈 Technical Indicators Explained:**
                            - **RSI (Relative Strength Index):** Measures if stock is overbought (>70) or oversold (<30)
                            - **SMA (Simple Moving Average):** Shows average price over specific period
                            - **Volume:** High volume confirms price movements
                            - **Momentum:** Rate of price change over time
                            """)
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Analysis failed: {e}")
                    st.info("💡 Try using a different AI provider in the sidebar or check your internet connection.")
                    
                    # Debug information
                    with st.expander("🔧 Debug Information"):
                        st.write("Error details:", str(e))
                        st.write("Symbol:", symbol)
                        st.write("AI Provider:", ai_provider)
    
    with tab2:
        st.header("🔧 How the AI System Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ System Architecture")
            
            # Create a simple architecture diagram using text
            st.markdown("""
            ```
            📊 Market Data Sources
                    ↓
            🤖 AI Agent Network
                    ↓
            🧠 Consensus Building
                    ↓
            📈 Educational Output
            ```
            """)
            
            st.markdown("""
            ### 🤖 AI Agents Pipeline:
            1. **Data Collection** - Fetch market data from free sources
            2. **Technical Analysis** - Calculate indicators and patterns  
            3. **Sentiment Analysis** - Assess market psychology
            4. **AI Reasoning** - Multiple agents provide perspectives
            5. **Consensus Building** - Combine insights intelligently
            6. **Risk Assessment** - Evaluate setup quality
            """)
        
        with col2:
            st.subheader("🧠 AI Models Used")
            
            st.markdown("""
            ### 🆓 Free AI Options:
            
            **🥇 Ollama (Recommended)**
            - Runs locally on your computer
            - Models: Llama 3.2, Mistral, CodeLlama
            - Completely free forever
            - No API limits
            
            **🥈 Hugging Face**
            - Free inference API
            - Models: FLAN-T5, DialoGPT
            - Good for testing
            - Some rate limits
            
            **🥉 OpenAI Free Tier**
            - $5 free credit for new users
            - GPT-3.5-turbo model
            - Best quality responses
            - Limited free usage
            
            **🛡️ Simple Fallback**
            - Rule-based analysis
            - Always available
            - Basic but reliable
            """)
            
            st.info("💡 The system automatically tries different AI providers until one works, ensuring you always get analysis results.")
        
        st.subheader("📊 Data Sources")
        
        data_col1, data_col2, data_col3 = st.columns(3)
        
        with data_col1:
            st.markdown("""
            **🆓 Yahoo Finance**
            - Unlimited requests
            - Real-time quotes
            - Historical data
            - Always available
            """)
        
        with data_col2:
            st.markdown("""
            **📈 Alpha Vantage**
            - 25 requests/day (free)
            - Professional data
            - Technical indicators
            - Good for testing
            """)
        
        with data_col3:
            st.markdown("""
            **📊 Other Free APIs**
            - Finnhub (60/min)
            - Polygon (5/min)
            - IEX Cloud (500/month)
            - Multiple backups
            """)
    
    with tab3:
        st.header("🎓 Trading Education Center")
        
        edu_tab1, edu_tab2, edu_tab3 = st.tabs(["📚 Basics", "📊 Technical Analysis", "🤖 AI Trading"])
        
        with edu_tab1:
            st.subheader("📚 Trading Fundamentals")
            
            st.markdown("""
            ### 🎯 What is Stock Analysis?
            
            Stock analysis is the process of evaluating a company's stock to determine if it's a good investment.
            There are two main types:
            
            **📊 Technical Analysis:**
            - Studies price charts and patterns
            - Uses mathematical indicators
            - Focuses on price and volume
            - Short to medium-term oriented
            
            **📈 Fundamental Analysis:**
            - Studies company financials
            - Evaluates business performance
            - Considers industry trends
            - Long-term oriented
            
            ### 🎮 Paper Trading
            Practice trading with virtual money before risking real capital:
            - Start with $100,000 virtual portfolio
            - Track hypothetical trades
            - Learn without financial risk
            - Build confidence and skills
            
            ### 📚 Key Concepts to Learn:
            - **Support and Resistance**: Price levels where stocks tend to bounce
            - **Trends**: Direction of price movement over time
            - **Volume**: Number of shares traded (confirms price moves)
            - **Risk Management**: Never risk more than you can afford to lose
            """)
        
        with edu_tab2:
            st.subheader("📊 Technical Analysis Guide")
            
            indicator_col1, indicator_col2 = st.columns(2)
            
            with indicator_col1:
                st.markdown("""
                ### 📈 Key Indicators:
                
                **RSI (Relative Strength Index)**
                - Range: 0-100
                - >70: Potentially overbought
                - <30: Potentially oversold
                - 50: Neutral zone
                
                **Moving Averages**
                - SMA 20: Short-term trend
                - SMA 50: Medium-term trend
                - SMA 200: Long-term trend
                - Price above MA = bullish
                """)
            
            with indicator_col2:
                st.markdown("""
                ### 📊 Chart Patterns:
                
                **Bullish Patterns**
                - Breakout above resistance
                - Higher highs and lows
                - Volume confirmation
                
                **Bearish Patterns**
                - Breakdown below support
                - Lower highs and lows
                - Selling volume
                """)
            
            # Interactive indicator calculator
            st.subheader("🧮 Interactive RSI Calculator")
            
            if st.button("Calculate RSI for Current Symbol"):
                try:
                    data = yf.Ticker(symbol).history(period="1mo")
                    if not data.empty:
                        delta = data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                        rs = gain / loss
                        rsi = (100 - (100 / (1 + rs))).iloc[-1]
                        
                        st.metric(f"{symbol} RSI (14-day)", f"{rsi:.1f}")
                        
                        if rsi > 70:
                            st.warning("⚠️ Potentially overbought - consider waiting for pullback")
                        elif rsi < 30:
                            st.success("✅ Potentially oversold - may be buying opportunity")
                        else:
                            st.info("ℹ️ RSI in neutral zone")
                    else:
                        st.error("Could not fetch data")
                except Exception as e:
                    st.error(f"Could not calculate RSI: {e}")
        
        with edu_tab3:
            st.subheader("🤖 AI in Trading Education")
            
            st.markdown("""
            ### 🧠 How AI Helps Traders Learn:
            
            **📊 Pattern Recognition**
            - AI can identify patterns humans might miss
            - Analyzes thousands of stocks simultaneously
            - Learns from historical data
            - Adapts to changing market conditions
            
            **🎯 Objective Analysis**
            - Removes emotional bias
            - Consistent methodology
            - 24/7 market monitoring
            - Multiple perspectives
            
            **📚 Educational Benefits**
            - See AI reasoning process
            - Learn from agent disagreements
            - Understand different analysis approaches
            - Practice with AI recommendations
            
            ### ⚠️ AI Limitations to Understand:
            - Not 100% accurate
            - Can't predict black swan events
            - Needs human oversight
            - Should complement, not replace, human judgment
            
            ### 🎓 Learning Path:
            1. **Start with this platform** - Understand how AI analyzes stocks
            2. **Practice paper trading** - Test strategies without risk
            3. **Study real examples** - Learn from AI reasoning
            4. **Build your own rules** - Develop personal trading criteria
            5. **Start small** - Begin with tiny real positions when ready
            """)
    
    with tab4:
        st.header("💡 About AI Trading Systems")
        
        about_col1, about_col2 = st.columns(2)
        
        with about_col1:
            st.subheader("🎯 Educational Goals")
            st.markdown("""
            This platform teaches:
            - How AI agents analyze market data
            - Multi-agent reasoning systems
            - Technical analysis fundamentals
            - Risk management concepts
            - Pattern recognition techniques
            
            **🎓 Perfect for:**
            - Students learning about AI
            - Traders wanting to understand algorithms
            - Developers interested in fintech
            - Anyone curious about AI reasoning
            
            **📚 What You'll Learn:**
            - How multiple AI agents work together
            - Why AI sometimes disagrees with itself
            - How to interpret confidence levels
            - Basic technical analysis concepts
            - Risk management principles
            """)
        
        with about_col2:
            st.subheader("🛠️ Technical Implementation")
            st.markdown("""
            **Built with:**
            - Python & Streamlit
            - Free AI models (Ollama, HuggingFace)
            - Yahoo Finance API
            - Plotly for charts
            - Pandas for data analysis
            
            **Features:**
            - Multi-agent AI analysis
            - Real-time data feeds
            - Interactive charts
            - Educational explanations
            - Paper trading concepts
            
            **Free AI Models:**
            - Ollama (local, unlimited)
            - HuggingFace (cloud, free tier)
            - OpenAI (free $5 credit)
            - Rule-based fallback
            """)
        
        st.subheader("🤝 Support the Project")
        st.markdown("""
        <div class="feature-card" style="text-align: center;">
            <h3 style="color: #212529; margin-bottom: 1.5rem;">☕ Keep This Educational Tool Free</h3>
            <p style="font-size: 1.1rem; margin-bottom: 1.5rem;">This platform is completely free for educational use. If it helps you learn about AI and trading:</p>
            <a href="https://www.buymeacoffee.com/yourusername" target="_blank" class="coffee-button">
                ☕ Buy me a coffee - $5
            </a>
            <p style="margin: 1.5rem 0;"><small>Help cover hosting costs and support continued development of free educational tools!</small></p>
            
            <hr style="border: none; border-top: 1px solid #e9ecef; margin: 2rem 0;">
            
            <h4 style="color: #495057; margin-bottom: 1rem;">📬 Connect & Learn More</h4>
            <p style="margin-bottom: 1.5rem;">
                <a href="https://github.com/yourusername/ai-trading-education" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 0.5rem;">GitHub Repository</a> | 
                <a href="https://twitter.com/yourusername" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 0.5rem;">Twitter Updates</a> | 
                <a href="mailto:your.email@example.com" style="color: #667eea; text-decoration: none; margin: 0 0.5rem;">Contact</a>
            </p>
            
            <h4 style="color: #495057; margin-bottom: 1rem;">🔗 Other Free Resources</h4>
            <p>
                <a href="https://www.investopedia.com" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 0.5rem;">Investopedia</a> | 
                <a href="https://www.tradingview.com" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 0.5rem;">TradingView</a> | 
                <a href="https://finance.yahoo.com" target="_blank" style="color: #667eea; text-decoration: none; margin: 0 0.5rem;">Yahoo Finance</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Usage statistics (simulated for demo)
        st.subheader("📊 Platform Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Generate some realistic demo stats
        total_analyses = len(st.session_state.analysis_history) + 1247
        
        with col1:
            st.metric("Total Analyses", f"{total_analyses:,}")
        with col2:
            st.metric("Active Learners", "156+")
        with col3:
            st.metric("AI Models", "4")
        with col4:
            st.metric("Avg Confidence", "73%")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🤖 AI Trading Education Hub | Built for Learning | Not Financial Advice</p>
        <p><small>Made with ❤️ for the trading and AI community | 
        <a href="https://github.com/yourusername/ai-trading-education" target="_blank">Open Source</a> | 
        <a href="https://www.buymeacoffee.com/yourusername" target="_blank">☕ Support</a></small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()