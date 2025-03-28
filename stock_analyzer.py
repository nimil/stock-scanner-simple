import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import logging
from openai import OpenAI
import akshare as ak
from base_analyzer import BaseAnalyzer

class StockAnalyzer(BaseAnalyzer):
    """股票分析器"""
    def __init__(self, initial_cash=1000000):
        super().__init__(initial_cash)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # 加载环境变量
        load_dotenv()
        
        # 设置 deepseek API
        self.deepseek_api_url = "https://api.deepseek.com"
        self.deepseek_api_key = os.getenv('DS_API_KEY')
        
        # 配置参数
        self.params = {
            'ma_periods': {'short': 5, 'medium': 20, 'long': 60},
            'rsi_period': 14,
            'bollinger_period': 20,
            'bollinger_std': 2,
            'volume_ma_period': 20,
            'atr_period': 14
        }
        
        # 初始化股票名称缓存
        self._stock_name_cache = {}  # 股票代码到名称的映射
        self._stock_name_cache_time = None
        self._cache_validity_hours = 24  # 缓存有效期（小时）
        
        # 初始化ETF名称缓存
        self._etf_name_cache = {}  # ETF代码到名称的映射
        self._etf_name_cache_time = None
        
        # 初始化指数名称映射
        self._index_names = {
            '000001': '上证指数',
            '399001': '深证成指',
            '399006': '创业板指',
            '000016': '上证50',
            '000300': '沪深300',
            '000905': '中证500',
            '000852': '中证1000'
        }
        
    def get_stock_data(self, stock_code, start_date=None, end_date=None):
        """获取股票数据"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        try:
            # 使用 akshare 获取股票数据
            df = ak.stock_zh_a_hist(symbol=stock_code, 
                                  start_date=start_date, 
                                  end_date=end_date,
                                  adjust="qfq")
            
            if df.empty:
                raise Exception(f"未获取到股票 {stock_code} 的数据")
            
            # 重命名列名以匹配分析需求
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume"
            })
            
            # 确保日期格式正确
            df['date'] = pd.to_datetime(df['date'])
            
            # 数据类型转换
            numeric_columns = ['open', 'close', 'high', 'low', 'volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            # 删除空值
            df = df.dropna()
            
            if df.empty:
                raise Exception(f"股票 {stock_code} 的数据处理后为空")
            
            return df.sort_values('date')
            
        except Exception as e:
            self.logger.error(f"获取股票数据失败: {str(e)}")
            raise Exception(f"获取股票数据失败: {str(e)}")
            
    def get_etf_data(self, etf_code, start_date=None, end_date=None):
        """获取ETF基金数据"""
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
                
            # 使用 akshare 获取ETF数据
            df = ak.fund_etf_hist_em(symbol=etf_code, 
                                    start_date=start_date, 
                                    end_date=end_date,
                                    adjust="qfq")
            
            if df.empty:
                raise Exception(f"未获取到ETF {etf_code} 的数据")
            
            # 重命名列名以匹配分析需求
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume"
            })
            
            # 确保日期格式正确
            df['date'] = pd.to_datetime(df['date'])
            
            # 数据类型转换
            numeric_columns = ['open', 'close', 'high', 'low', 'volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            # 删除空值
            df = df.dropna()
            
            if df.empty:
                raise Exception(f"ETF {etf_code} 的数据处理后为空")
            
            return df.sort_values('date')
            
        except Exception as e:
            self.logger.error(f"获取ETF数据失败: {str(e)}")
            raise Exception(f"获取ETF数据失败: {str(e)}")
            
    def calculate_ema(self, series, period):
        """计算指数移动平均线"""
        return series.ewm(span=period, adjust=False).mean()
        
    def calculate_rsi(self, series, period):
        """计算RSI指标"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def calculate_macd(self, series):
        """计算MACD指标"""
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist
        
    def calculate_bollinger_bands(self, series, period, std_dev):
        """计算布林带"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
        
    def calculate_atr(self, df, period):
        """计算ATR指标"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
        
    def calculate_indicators(self, df):
        """计算技术指标"""
        try:
            # 计算移动平均线
            df['MA5'] = self.calculate_ema(df['close'], self.params['ma_periods']['short'])
            df['MA20'] = self.calculate_ema(df['close'], self.params['ma_periods']['medium'])
            df['MA60'] = self.calculate_ema(df['close'], self.params['ma_periods']['long'])
            
            # 计算RSI
            df['RSI'] = self.calculate_rsi(df['close'], self.params['rsi_period'])
            
            # 计算MACD
            df['MACD'], df['Signal'], df['MACD_hist'] = self.calculate_macd(df['close'])
            
            # 计算布林带
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(
                df['close'],
                self.params['bollinger_period'],
                self.params['bollinger_std']
            )
            
            # 成交量分析
            df['Volume_MA'] = df['volume'].rolling(window=self.params['volume_ma_period']).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA']
            
            # 计算ATR和波动率
            df['ATR'] = self.calculate_atr(df, self.params['atr_period'])
            df['Volatility'] = df['ATR'] / df['close'] * 100
            
            # 动量指标
            df['ROC'] = df['close'].pct_change(periods=10) * 100
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算技术指标时出错: {str(e)}")
            raise
            
    def calculate_score(self, df):
        """计算股票评分"""
        try:
            score = 0
            latest = df.iloc[-1]
            
            # 趋势得分 (30分)
            if latest['MA5'] > latest['MA20']:
                score += 15
            if latest['MA20'] > latest['MA60']:
                score += 15
                
            # RSI得分 (20分)
            if 30 <= latest['RSI'] <= 70:
                score += 20
            elif latest['RSI'] < 30:  # 超卖
                score += 15
                
            # MACD得分 (20分)
            if latest['MACD'] > latest['Signal']:
                score += 20
                
            # 成交量得分 (30分)
            if latest['Volume_Ratio'] > 1.5:
                score += 30
            elif latest['Volume_Ratio'] > 1:
                score += 15
                
            return score
            
        except Exception as e:
            self.logger.error(f"计算评分时出错: {str(e)}")
            raise
            
    def calculate_etf_score(self, df):
        """计算ETF评分"""
        try:
            score = 60  # 基础分
            latest = df.iloc[-1]
            
            # 趋势得分
            if latest['MA5'] > latest['MA20']:
                score += 10
            if latest['MA20'] > latest['MA60']:
                score += 10
                
            # RSI得分
            rsi = latest['RSI']
            if 40 <= rsi <= 60:
                score += 10
            elif 30 <= rsi <= 70:
                score += 5
                
            # MACD得分
            if latest['MACD'] > latest['Signal']:
                score += 10
                
            # 成交量分析
            if latest['Volume_Ratio'] > 1.2:
                score += 10
                
            return min(100, score)
            
        except Exception as e:
            self.logger.error(f"计算ETF评分时出错: {str(e)}")
            raise
            
    def get_recommendation(self, score):
        """根据得分给出建议"""
        if score >= 80:
            return '强烈推荐买入'
        elif score >= 60:
            return '建议买入'
        elif score >= 40:
            return '观望'
        elif score >= 20:
            return '建议卖出'
        else:
            return '强烈建议卖出'
            
    def get_etf_recommendation(self, score):
        """根据得分给出ETF建议"""
        if score >= 80:
            return '建议关注'
        elif score >= 60:
            return '可以观望'
        else:
            return '建议谨慎'
            
    def get_ai_analysis(self, df, code, asset_type="股票"):
        """使用 deepSeek 进行 AI 分析"""
        try:
            recent_data = df.tail(14).to_dict('records')
            
            technical_summary = {
                'trend': 'upward' if df.iloc[-1]['MA5'] > df.iloc[-1]['MA20'] else 'downward',
                'volatility': f"{df.iloc[-1]['Volatility']:.2f}%",
                'volume_trend': 'increasing' if df.iloc[-1]['Volume_Ratio'] > 1 else 'decreasing',
                'rsi_level': df.iloc[-1]['RSI']
            }
            
            prompt = f"""
            分析{asset_type} {code}：

            技术指标概要：
            {technical_summary}
            
            近14日交易数据：
            {recent_data}
            
            请提供：
            1. 趋势分析（包含支撑位和压力位）
            2. 成交量分析及其含义
            3. 风险评估（包含波动率分析）
            4. 短期和中期目标价位
            5. 关键技术位分析
            6. 具体交易建议（包含止损位）
            
            请基于技术指标和市场动态进行分析，给出具体数据支持。
            """
            client = OpenAI(api_key=self.deepseek_api_key,base_url=self.deepseek_api_url)
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
        
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"AI 分析发生错误: {str(e)}")
            return "AI 分析过程中发生错误"
            
    def _get_stock_info(self, stock_code: str) -> str:
        """获取股票名称，使用缓存机制"""
        current_time = datetime.now()
        
        # 如果缓存不存在或已过期，更新缓存
        if (self._stock_name_cache_time is None or 
            (current_time - self._stock_name_cache_time).total_seconds() > self._cache_validity_hours * 3600):
            
            try:
                # 获取所有股票信息
                stock_info = ak.stock_zh_a_spot_em()
                # 只缓存代码和名称的映射
                self._stock_name_cache = dict(zip(stock_info['代码'], stock_info['名称']))
                self._stock_name_cache_time = current_time
                self.logger.info("已更新股票名称缓存")
            except Exception as e:
                self.logger.error(f"更新股票名称缓存失败: {str(e)}")
                raise
        
        try:
            if stock_code not in self._stock_name_cache:
                raise KeyError(f"未找到股票代码: {stock_code}")
            return self._stock_name_cache[stock_code]
        except Exception as e:
            self.logger.error(f"获取股票名称失败: {str(e)}")
            raise
            
    def _get_etf_info(self, etf_code: str) -> str:
        """获取ETF名称，使用缓存机制"""
        current_time = datetime.now()
        
        # 如果缓存不存在或已过期，更新缓存
        if (self._etf_name_cache_time is None or 
            (current_time - self._etf_name_cache_time).total_seconds() > self._cache_validity_hours * 3600):
            
            try:
                # 获取所有ETF基金信息
                etf_info = ak.fund_name_em()
                # 只缓存代码和名称的映射
                self._etf_name_cache = dict(zip(etf_info['基金代码'], etf_info['基金简称']))
                self._etf_name_cache_time = current_time
                self.logger.info("已更新ETF名称缓存")
            except Exception as e:
                self.logger.error(f"更新ETF名称缓存失败: {str(e)}")
                raise
        
        try:
            if etf_code not in self._etf_name_cache:
                self.logger.warning(f"未找到ETF代码: {etf_code}, 使用默认名称")
                return f"ETF基金({etf_code})"
            return self._etf_name_cache[etf_code]
        except Exception as e:
            self.logger.error(f"获取ETF名称失败: {str(e)}")
            return f"ETF基金({etf_code})"  # 返回默认名称
            
    def analyze_stock(self, stock_code: str) -> Dict:
        """分析单只股票"""
        try:
            # 获取股票名称（使用缓存机制）
            stock_name = self._get_stock_info(stock_code)
            
            # 获取历史数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
            # 获取股票数据
            df = self.get_stock_data(stock_code, start_date, end_date)
            
            # 计算技术指标
            df = self.calculate_indicators(df)
            
            # 评分系统
            score = self.calculate_score(df)
            
            # 获取最新数据
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 生成报告（保持原有格式）
            report = {
                'stock_code': stock_code,
                'stock_name': stock_name,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'price': latest['close'],
                'price_change': (latest['close'] - prev['close']) / prev['close'] * 100,
                'ma_trend': 'UP' if latest['MA5'] > latest['MA20'] else 'DOWN',
                'rsi': latest['RSI'],
                'macd_signal': 'BUY' if latest['MACD'] > latest['Signal'] else 'SELL',
                'volume_status': 'HIGH' if latest['Volume_Ratio'] > 1.5 else 'NORMAL',
                'score': score,
                'recommendation': self.get_recommendation(score),
                'ai_analysis': self.get_ai_analysis(df, stock_code, "股票")
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"分析股票时出错: {str(e)}")
            raise
            
    def analyze_etf(self, etf_code: str) -> Dict:
        """分析ETF基金"""
        try:
            # 获取ETF名称
            etf_name = self._get_etf_info(etf_code)
            
            # 获取ETF数据
            df = self.get_etf_data(etf_code)
            
            # 计算技术指标
            df = self.calculate_indicators(df)
            
            # 评分系统
            score = self.calculate_etf_score(df)
            
            # 获取最新数据
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 生成报告
            report = {
                'stock_code': etf_code,
                'stock_name': etf_name,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'price': float(latest['close']),
                'price_change': float((latest['close'] - prev['close']) / prev['close'] * 100),
                'ma_trend': 'UP' if latest['MA5'] > latest['MA20'] else 'DOWN',
                'rsi': float(latest['RSI']),
                'macd_signal': 'BUY' if latest['MACD'] > latest['Signal'] else 'SELL',
                'volume_status': '放量' if latest['Volume_Ratio'] > 1.2 else '缩量',
                'score': score,
                'recommendation': self.get_etf_recommendation(score),
                'ai_analysis': self.get_ai_analysis(df, etf_code, "ETF基金")
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"分析ETF时出错: {str(e)}")
            raise
            
    def scan_market(self, stock_list, min_score=60):
        """扫描市场，寻找符合条件的股票"""
        recommendations = []
        
        for stock_code in stock_list:
            try:
                report = self.analyze_stock(stock_code)
                if report['score'] >= min_score:
                    recommendations.append(report)
            except Exception as e:
                self.logger.error(f"分析股票 {stock_code} 时出错: {str(e)}")
                continue
                
        # 按得分排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations
    
    def scan_etf_market(self, etf_list, min_score=60):
        """扫描ETF市场，寻找符合条件的ETF"""
        recommendations = []
        
        for etf_code in etf_list:
            try:
                report = self.analyze_etf(etf_code)
                if report['score'] >= min_score:
                    recommendations.append(report)
            except Exception as e:
                self.logger.error(f"分析ETF {etf_code} 时出错: {str(e)}")
                continue
                
        # 按得分排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations

    def get_index_data(self, index_code, start_date=None, end_date=None):
        """获取指数数据"""
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
                
            # 使用 akshare 获取指数数据
            df = ak.index_zh_a_hist(symbol=index_code, 
                                  start_date=start_date, 
                                  end_date=end_date)
            
            if df.empty:
                raise Exception(f"未获取到指数 {index_code} 的数据")
            
            # 重命名列名以匹配分析需求
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume"
            })
            
            # 确保日期格式正确
            df['date'] = pd.to_datetime(df['date'])
            
            # 数据类型转换
            numeric_columns = ['open', 'close', 'high', 'low', 'volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            # 删除空值
            df = df.dropna()
            
            if df.empty:
                raise Exception(f"指数 {index_code} 的数据处理后为空")
            
            return df.sort_values('date')
            
        except Exception as e:
            self.logger.error(f"获取指数数据失败: {str(e)}")
            raise Exception(f"获取指数数据失败: {str(e)}")

    def calculate_index_score(self, df):
        """计算指数评分"""
        try:
            score = 50  # 基础分
            latest = df.iloc[-1]
            
            # 趋势得分 (20分)
            if latest['MA5'] > latest['MA20']:
                score += 10
            if latest['MA20'] > latest['MA60']:
                score += 10
                
            # RSI得分 (10分)
            rsi = latest['RSI']
            if 40 <= rsi <= 60:
                score += 10
            elif 30 <= rsi <= 70:
                score += 5
                
            # MACD得分 (10分)
            if latest['MACD'] > latest['Signal']:
                score += 10
                
            # 成交量分析 (10分)
            if latest['Volume_Ratio'] > 1.2:
                score += 10
                
            return min(100, score)
            
        except Exception as e:
            self.logger.error(f"计算指数评分时出错: {str(e)}")
            raise

    def get_index_recommendation(self, score):
        """根据得分给出指数建议"""
        if score >= 80:
            return '市场强势，可积极参与'
        elif score >= 60:
            return '市场偏强，可择机参与'
        elif score >= 40:
            return '市场震荡，保持谨慎'
        else:
            return '市场偏弱，建议观望'

    def analyze_index(self, index_code: str) -> Dict:
        """分析指数"""
        try:
            # 获取指数名称
            index_name = self._index_names.get(index_code, f"指数({index_code})")
            
            # 获取指数数据
            df = self.get_index_data(index_code)
            
            # 计算技术指标
            df = self.calculate_indicators(df)
            
            # 评分系统
            score = self.calculate_index_score(df)
            
            # 获取最新数据
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 生成报告
            report = {
                'stock_code': index_code,
                'stock_name': index_name,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'price': float(latest['close']),
                'price_change': float((latest['close'] - prev['close']) / prev['close'] * 100),
                'ma_trend': 'UP' if latest['MA5'] > latest['MA20'] else 'DOWN',
                'rsi': float(latest['RSI']),
                'macd_signal': 'BUY' if latest['MACD'] > latest['Signal'] else 'SELL',
                'volume_status': '放量' if latest['Volume_Ratio'] > 1.2 else '缩量',
                'score': score,
                'recommendation': self.get_index_recommendation(score),
                'ai_analysis': self.get_ai_analysis(df, index_code, "指数")
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"分析指数时出错: {str(e)}")
            raise 