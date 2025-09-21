"""
Free AI Trading System - Zero Investment Prototype
Uses completely free AI models and data sources
File: free_ai_trading_system.py
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
import pandas as pd
import numpy as np
import os

# Free AI models and libraries
try:
    import openai  # Free tier: $5 credit for new accounts
except ImportError:
    openai = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    torch = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading signal from AI analysis"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    reasoning: str
    agent_name: str
    timestamp: datetime
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None

class OllamaProvider:
    """Free local AI using Ollama - completely free"""
    
    def __init__(self):
        self.client = None
        self.model = "llama3.2:3b"  # Fast, good quality model
    
    def is_available(self) -> bool:
        """Check if Ollama is running locally"""
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response using Ollama"""
        try:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return None

class HuggingFaceProvider:
    """Free Hugging Face Inference API"""
    
    def __init__(self):
        self.models = [
            "google/flan-t5-large",
            "microsoft/DialoGPT-medium",
            "facebook/blenderbot-400M-distill"
        ]
        self.current_model = "google/flan-t5-large"
        self.client = InferenceClient() if InferenceClient else None
    
    def is_available(self) -> bool:
        return self.client is not None
    
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Use HuggingFace free inference"""
        if not self.client:
            return None
            
        try:
            # Format prompt for instruction-following models
            full_prompt = f"Instruction: {system_prompt}\n\nInput: {prompt}\n\nOutput:"
            
            response = self.client.text_generation(
                prompt=full_prompt,
                model=self.current_model,
                max_new_tokens=500,
                temperature=0.3,
                return_full_text=False
            )
            
            return response if response else "No response generated"
            
        except Exception as e:
            logger.error(f"HuggingFace error: {e}")
            return None

class OpenAIFreeProvider:
    """OpenAI with free $5 credit for new accounts"""
    
    def __init__(self):
        self.client = openai.AsyncOpenAI() if openai else None
        self.model = "gpt-3.5-turbo"  # Cheapest option
    
    def is_available(self) -> bool:
        """Check if OpenAI API key is set"""
        return bool(self.client and os.getenv('OPENAI_API_KEY'))
    
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Use OpenAI free tier"""
        if not self.client:
            return None
            
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,  # Keep costs low
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return None

class SimpleAIProvider:
    """Fallback AI using simple rules-based analysis"""
    
    def is_available(self) -> bool:
        return True
    
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Simple rule-based analysis"""
        try:
            # Extract key data from prompt
            lines = prompt.split('\n')
            analysis = {}
            
            for line in lines:
                if 'Current Price:' in line:
                    analysis['price'] = line
                elif 'Daily Change:' in line:
                    analysis['change'] = line
                elif 'RSI:' in line:
                    analysis['rsi'] = float(line.split(':')[1].strip()) if ':' in line else 50
                elif 'vs 20-day SMA:' in line:
                    analysis['sma20'] = line
            
            # Simple rule-based logic
            rsi = analysis.get('rsi', 50)
            
            if rsi > 70:
                action = "SELL"
                confidence = 65
                reasoning = f"RSI at {rsi:.1f} indicates overbought conditions. Consider taking profits or waiting for pullback."
            elif rsi < 30:
                action = "BUY"
                confidence = 70
                reasoning = f"RSI at {rsi:.1f} suggests oversold conditions. Potential buying opportunity if other factors align."
            else:
                action = "HOLD"
                confidence = 55
                reasoning = f"RSI at {rsi:.1f} in neutral zone. No clear signal. Monitor for trend changes."
            
            return f"{action}|{confidence}|{reasoning}|None|None"
            
        except Exception as e:
            return "HOLD|50|Simple analysis unavailable due to data parsing error|None|None"

class FreeAIProvider:
    """Manages multiple free AI providers"""
    
    def __init__(self):
        self.providers = {
            'ollama': OllamaProvider(),
            'huggingface': HuggingFaceProvider(), 
            'openai_free': OpenAIFreeProvider(),
            'simple': SimpleAIProvider()
        }
        self.current_provider = 'ollama'  # Default to fully free option
    
    async def analyze(self, prompt: str, system_prompt: str = "") -> str:
        """Try providers in order until one works"""
        for provider_name, provider in self.providers.items():
            try:
                if provider.is_available():
                    response = await provider.generate(prompt, system_prompt)
                    if response:
                        logger.info(f"Used {provider_name} for analysis")
                        return response
            except Exception as e:
                logger.warning(f"{provider_name} failed: {e}")
                continue
        
        return "HOLD|50|Analysis unavailable - no AI providers working|None|None"

class FreeDataProvider:
    """Free market data sources"""
    
    def __init__(self):
        self.sources = {
            'yahoo': self._get_yahoo_data,
            'alphavantage_free': self._get_alphavantage_free,
            'finnhub_free': self._get_finnhub_free
        }
    
    def get_market_data(self, symbol: str, period: str = "3mo") -> Optional[pd.DataFrame]:
        """Try free data sources in order"""
        for source_name, source_func in self.sources.items():
            try:
                data = source_func(symbol, period)
                if data is not None and not data.empty:
                    logger.info(f"Got data from {source_name}")
                    return data
            except Exception as e:
                logger.warning(f"{source_name} failed: {e}")
                continue
        
        return None
    
    def _get_yahoo_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Yahoo Finance - completely free"""
        if not yf:
            return None
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period)
    
    def _get_alphavantage_free(self, symbol: str, period: str) -> pd.DataFrame:
        """Alpha Vantage free tier - 25 requests/day"""
        api_key = os.getenv('ALPHA_VANTAGE_KEY')
        if not api_key:
            return None
            
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': api_key,
            'outputsize': 'compact'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df.sort_index()
    
    def _get_finnhub_free(self, symbol: str, period: str) -> pd.DataFrame:
        """Finnhub free tier - 60 requests/minute"""
        api_key = os.getenv('FINNHUB_API_KEY')
        if not api_key:
            return None
        
        # Calculate date range
        end_date = datetime.now()
        if period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=365)
        
        url = "https://finnhub.io/api/v1/stock/candle"
        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': int(start_date.timestamp()),
            'to': int(end_date.timestamp()),
            'token': api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get('s') != 'ok':
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'Open': data['o'],
            'High': data['h'], 
            'Low': data['l'],
            'Close': data['c'],
            'Volume': data['v']
        })
        df.index = pd.to_datetime(data['t'], unit='s')
        return df

class FreeTradingAgent:
    """Trading agent using free AI models"""
    
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.ai_provider = FreeAIProvider()
        self.data_provider = FreeDataProvider()
    
    async def analyze(self, symbol: str, context: Dict = None) -> TradingSignal:
        """Analyze symbol using free AI and data"""
        
        # Get free market data
        data = self.data_provider.get_market_data(symbol)
        if data is None or data.empty:
            return TradingSignal(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                reasoning="No market data available",
                agent_name=self.name,
                timestamp=datetime.now()
            )
        
        # Calculate basic indicators
        indicators = self._calculate_indicators(data)
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(symbol, data, indicators, context or {})
        
        # Get AI response
        response = await self.ai_provider.analyze(prompt, self.system_prompt)
        
        # Parse response
        signal = self._parse_response(symbol, response)
        
        return signal
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate free technical indicators"""
        indicators = {}
        
        try:
            # Moving averages
            indicators['sma_20'] = data['Close'].rolling(20).mean().iloc[-1]
            indicators['sma_50'] = data['Close'].rolling(50).mean().iloc[-1]
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Simple momentum
            indicators['momentum_5d'] = ((data['Close'].iloc[-1] / data['Close'].iloc[-6]) - 1) * 100
            indicators['momentum_20d'] = ((data['Close'].iloc[-1] / data['Close'].iloc[-21]) - 1) * 100
            
            # Volume analysis
            indicators['volume_ratio'] = data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            # Set default values
            indicators = {
                'sma_20': data['Close'].iloc[-1] if not data.empty else 100,
                'sma_50': data['Close'].iloc[-1] if not data.empty else 100,
                'rsi': 50,
                'momentum_5d': 0,
                'momentum_20d': 0,
                'volume_ratio': 1
            }
        
        return indicators
    
    def _create_analysis_prompt(self, symbol: str, data: pd.DataFrame, indicators: Dict, context: Dict) -> str:
        """Create analysis prompt for AI"""
        current_price = data['Close'].iloc[-1]
        daily_change = ((current_price / data['Close'].iloc[-2]) - 1) * 100
        
        prompt = f"""
Analyze {symbol} for trading opportunities:

PRICE DATA:
- Current Price: ${current_price:.2f}
- Daily Change: {daily_change:+.2f}%
- 5-day momentum: {indicators.get('momentum_5d', 0):.2f}%
- 20-day momentum: {indicators.get('momentum_20d', 0):.2f}%

TECHNICAL INDICATORS:
- RSI: {indicators.get('rsi', 50):.1f}
- Price vs 20-day SMA: {((current_price / indicators.get('sma_20', current_price)) - 1) * 100:+.1f}%
- Price vs 50-day SMA: {((current_price / indicators.get('sma_50', current_price)) - 1) * 100:+.1f}%
- Volume ratio vs average: {indicators.get('volume_ratio', 1):.1f}x

Provide trading recommendation with:
1. Action (BUY/SELL/HOLD)
2. Confidence (0-100%)
3. Brief reasoning
4. Price target (if bullish)
5. Stop loss level

Format as: ACTION|CONFIDENCE|REASONING|TARGET|STOP
Example: BUY|75|Strong momentum with volume|195.00|180.00
"""
        return prompt
    
    def _parse_response(self, symbol: str, response: str) -> TradingSignal:
        """Parse AI response into trading signal"""
        try:
            # Try to parse structured response
            if '|' in response:
                parts = response.split('|')
                if len(parts) >= 3:
                    action = parts[0].strip().upper()
                    confidence = float(parts[1].strip()) / 100
                    reasoning = parts[2].strip()
                    price_target = float(parts[3].strip()) if len(parts) > 3 and parts[3].strip() and parts[3].strip() != 'None' else None
                    stop_loss = float(parts[4].strip()) if len(parts) > 4 and parts[4].strip() and parts[4].strip() != 'None' else None
                    
                    return TradingSignal(
                        symbol=symbol,
                        action=action if action in ['BUY', 'SELL', 'HOLD'] else 'HOLD',
                        confidence=min(max(confidence, 0), 1),
                        reasoning=reasoning,
                        agent_name=self.name,
                        timestamp=datetime.now(),
                        price_target=price_target,
                        stop_loss=stop_loss
                    )
            
            # Fallback: parse free-form response
            action = 'HOLD'
            confidence = 0.5
            
            response_lower = response.lower()
            if 'buy' in response_lower and 'sell' not in response_lower:
                action = 'BUY'
                confidence = 0.6
            elif 'sell' in response_lower and 'buy' not in response_lower:
                action = 'SELL'
                confidence = 0.6
            
            return TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                reasoning=response[:200] + '...' if len(response) > 200 else response,
                agent_name=self.name,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return TradingSignal(
                symbol=symbol,
                action='HOLD',
                confidence=0.0,
                reasoning=f"Failed to parse AI response: {response[:100]}...",
                agent_name=self.name,
                timestamp=datetime.now()
            )

class FreeTradingSystem:
    """Complete free trading system"""
    
    def __init__(self):
        self.market_agent = FreeTradingAgent(
            "Market Analyzer",
            "You are a trading analyst. Analyze market data and provide buy/sell/hold recommendations with confidence levels and reasoning. Focus on technical indicators, price patterns, and momentum."
        )
        
        self.sentiment_agent = FreeTradingAgent(
            "Sentiment Analyzer", 
            "You are a market sentiment analyst. Focus on momentum, volume patterns, and technical indicators to gauge market sentiment. Consider overbought/oversold conditions and trend strength."
        )
    
    async def analyze_symbol(self, symbol: str, context: Dict = None) -> Dict:
        """Comprehensive free analysis"""
        
        print(f"🔍 Analyzing {symbol} using free AI models...")
        
        # Run agents in parallel
        market_task = self.market_agent.analyze(symbol, context)
        sentiment_task = self.sentiment_agent.analyze(symbol, context)
        
        try:
            market_signal, sentiment_signal = await asyncio.gather(market_task, sentiment_task)
            
            # Simple consensus logic
            signals = [market_signal, sentiment_signal]
            
            # Weight by confidence
            total_confidence = sum(s.confidence for s in signals)
            if total_confidence > 0:
                buy_weight = sum(s.confidence for s in signals if s.action == 'BUY')
                sell_weight = sum(s.confidence for s in signals if s.action == 'SELL')
                
                if buy_weight > sell_weight and buy_weight > 0.5:
                    consensus_action = 'BUY'
                    consensus_confidence = buy_weight / len(signals)
                elif sell_weight > buy_weight and sell_weight > 0.5:
                    consensus_action = 'SELL'
                    consensus_confidence = sell_weight / len(signals)
                else:
                    consensus_action = 'HOLD'
                    consensus_confidence = 0.5
            else:
                consensus_action = 'HOLD'
                consensus_confidence = 0.0
            
            return {
                'symbol': symbol,
                'consensus': {
                    'action': consensus_action,
                    'confidence': consensus_confidence,
                    'reasoning': f"Market Agent: {market_signal.action} ({market_signal.confidence:.1%}), Sentiment Agent: {sentiment_signal.action} ({sentiment_signal.confidence:.1%})"
                },
                'signals': [market_signal, sentiment_signal],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Test function
async def test_system():
    """Test the free trading system"""
    system = FreeTradingSystem()
    
    test_symbols = ['AAPL', 'TSLA']
    
    for symbol in test_symbols:
        print(f"\n{'='*50}")
        print(f"Testing {symbol}")
        print('='*50)
        
        result = await system.analyze_symbol(symbol)
        
        if 'error' not in result:
            consensus = result['consensus']
            signals = result['signals']
            
            print(f"🎯 Consensus: {consensus['action']} ({consensus['confidence']:.1%})")
            print(f"💭 Reasoning: {consensus['reasoning']}")
            
            print(f"\n🤖 Individual Agent Analysis:")
            for signal in signals:
                print(f"  • {signal.agent_name}: {signal.action} ({signal.confidence:.1%})")
                print(f"    {signal.reasoning[:100]}...")
        else:
            print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    print("🚀 Free AI Trading System Test")
    print("=" * 50)
    asyncio.run(test_system())