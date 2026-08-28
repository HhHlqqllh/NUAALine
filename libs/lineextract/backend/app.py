from flask import Flask, request, jsonify, Response

import time
import lineextract

app = Flask(__name__)

@app.route('/api/save_path', methods=['GET'])
def save_path():
    path = request.args.get('path')
    print(path)
    try:
        result = lineextract.process_path(path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/progress')
def progress():
    print("收到 /progress 请求")
    def generate():
        print("开始生成 SSE 数据")
        # 立即发送初始数据
        yield "data: 0.00\n\n"
        last_progress = 0
        timeout = 6000  # 超时时间（秒）
        start_time = time.time()
        while last_progress < 100:
            if time.time() - start_time > timeout:
                print("进度推送超时")
                yield "data: 100.00\n\n"
                break
            current_progress = lineextract.progress_value
            #print(f"检查进度: {current_progress:.2f}")
            if current_progress != last_progress:
                #print(f"推送进度: {current_progress:.2f}")
                yield f"data: {current_progress:.2f}\n\n"
                last_progress = current_progress
            time.sleep(0.1)
        #print("推送最终进度: 100.00")
        yield "data: 100.00\n\n"
    return Response(generate(), mimetype='text/event-stream', headers={
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    })


if __name__ == '__main__':
    app.run(debug=False)