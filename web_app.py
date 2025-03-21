from flask import Flask, request, jsonify, render_template
from stock_analyzer import StockAnalyzer
import os

app = Flask(__name__)
analyzer = StockAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        stock_code = data.get('stock_code')
        asset_type = data.get('asset_type', 'stock')  # 默认为股票类型
        
        if not stock_code:
            return jsonify({'error': '请提供交易代码'}), 400

        if asset_type == 'etf':
            # 调用StockAnalyzer中处理ETF的方法
            result = analyzer.analyze_etf(stock_code)
        else:
            result = analyzer.analyze_stock(stock_code)
            
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    try:
        data = request.json
        stock_list = data.get('stock_list', [])
        asset_type = data.get('asset_type', 'stock')  # 默认为股票类型
        
        if not stock_list:
            return jsonify({'error': '请提供代码列表'}), 400

        if asset_type == 'etf':
            # 假设我们有批量分析ETF的方法
            results = analyzer.scan_etf_market(stock_list)
        else:
            results = analyzer.scan_market(stock_list)
            
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8443)
