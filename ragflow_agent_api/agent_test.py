from ragflow_sdk import RAGFlow, Agent
from flask import Flask, request, jsonify
import json
from collections import OrderedDict
from flask import make_response

app = Flask(__name__)
def ragflow_agent_query(api_key: str,
                        base_url: str,
                        agent_id: str,
                        question: str) -> str:
    """非交互式调用RAGFlow Agent，获取指定问题的回答"""
    # 初始化RAGFlow客户端
    rag_object = RAGFlow(api_key=api_key, base_url=base_url)

    # 获取指定Agent
    try:
        agent = rag_object.list_agents(id=agent_id)[0]
    except (IndexError, Exception) as e:
        print(f"获取Agent失败: {str(e)}")
        return ""

    # 创建会话
    try:
        session = agent.create_session()
    except Exception as e:
        print(f"创建会话失败: {str(e)}")
        return ""

    # 处理问题并获取回答
    try:
        full_answer = ""
        for ans in session.ask(question, stream=True):
            full_answer = ans.content  # 累积完整回答
        return full_answer
    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        return ""


@app.route('/get_agent_answer', methods=['POST'])
def get_agent_answer():
    try:
        # 获取请求中的JSON数据
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求数据不能为空'}), 400

        # 提取问题参数（假设请求格式为 {"question": "具体问题内容"}）
        question = json.dumps(data, ensure_ascii=False)

        # 调用Agent获取回答
        answer = ragflow_agent_query(
            api_key= "",
            base_url= "",
            agent_id= "",
            question=question
        )

        if not answer:
            return jsonify({'error': '未能获取有效的回答'}), 500

        try:
            answer_json = json.loads(answer, object_pairs_hook=OrderedDict)  # 强制保留插入顺序
        except json.JSONDecodeError:
            answer_json = {"answer": answer}

        response_data = OrderedDict()
        response_data["工艺"] = answer_json

        response_json = json.dumps(
            response_data,
            ensure_ascii=False,
            sort_keys=False,  # 禁止排序
            indent=2
        )
        return make_response(response_json, 200, {"Content-Type": "application/json"})

    except Exception as e:
        return jsonify({'error': f'调用Agent服务失败: {str(e)}'}), 500


if __name__ == '__main__':
    # 启动Flask服务（默认端口5000，debug模式仅用于开发环境）
    app.run(host='0.0.0.0', port=5001, debug=False)


#curl -X POST http://localhost:5001/get_agent_answer -H "Content-Type: application/json" -d "{\"question\": {\"零件名称\": \"齿轮\", \"材料\": \"20CrMnTi\", \"精度等级\": \"IT6\", \"表面粗糙度\": \"Ra0.8\"}}"
