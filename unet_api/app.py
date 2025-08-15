from flask import Flask, request, jsonify
import os
import datetime
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from net import UNet
from utils import keep_image_size_open_rgb
from torchvision import transforms
import base64
from openai import OpenAI
import tempfile
import re
import pandas as pd
import json

# ========= 初始化模型 =========
num_classes = 3
weights_path = 'params/unet-tuzhi0730.pth'
net = UNet(num_classes)
net.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
net.eval()

# ========= 加载Excel中的工艺数据，包括工艺名称、计费方式和工艺成本 =========
EXCEL_FILE_PATH = "process_costs.xlsx"  # 确保该文件存在于项目根目录

# ========= Flask app =========
app = Flask(__name__)

# ========= 图像预处理变换 =========
FIXED_SIZE = (224, 224)
image_transform = transforms.Compose([
    transforms.Resize(FIXED_SIZE),
    transforms.ToTensor()
])

# ========= 图像处理函数 =========
def expand_mask_until_uniform(mask, max_iters=10):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for _ in range(max_iters):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        need_expand = False
        for cnt in contours:
            for point in cnt:
                x, y = point[0]
                neighborhood = mask[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
                if neighborhood.size == 0:
                    continue
                if not (np.all(neighborhood == 0) or np.all(neighborhood == 1)):
                    need_expand = True
                    break
            if need_expand:
                break
        if not need_expand:
            break
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def smooth_mask_with_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smooth_mask = np.zeros_like(mask)
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.fillPoly(smooth_mask, [approx], 1)
    return smooth_mask

def process_image(image_path, output_dir):
    img = keep_image_size_open_rgb(image_path)
    orig_w, orig_h = img.size
    img_data = image_transform(img).unsqueeze(0)

    with torch.no_grad():
        out = net(img_data)
        out_upsampled = F.interpolate(out, size=(orig_h, orig_w), mode='nearest')
        pred = torch.argmax(out_upsampled, dim=1).squeeze().cpu().numpy()

    original = np.array(img)
    result_images = []

    for cls_id in [1, 2]:
        mask = (pred == cls_id).astype(np.uint8)
        if mask.sum() == 0:
            continue
        mask_refined = expand_mask_until_uniform(mask)
        mask_smooth = smooth_mask_with_polygon(mask_refined)

        ys, xs = np.where(mask_smooth == 1)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        result_img = original.copy()
        white_background = np.ones_like(result_img) * 255
        mask_3ch = np.stack([mask_smooth] * 3, axis=-1)
        result_img = np.where(mask_3ch == 1, result_img, white_background)
        cropped = result_img[y_min:y_max + 1, x_min:x_max + 1]

        save_name = f"class{cls_id}_crop_whitebg.png"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        result_images.append(save_path)

    return result_images

def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ========= 整理PIC和BOM信息的大模型 =========
def summarize_with_deepseek(text_to_summarize):
    try:
        from openai import OpenAI
        client = OpenAI(api_key="", base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一位专业的工业图纸分析专家，擅长从识别结果中整理技术要求。"
                        "你需要将给定的信息明确重新排列为以下五个技术类别，并以嵌套 JSON 格式输出："
                        "1. 零件几何参数，2. 精度与公差要求，3. 表面与加工要求，4. 材料与热处理，5. 其他。"
                        "必须返回所有出现的信息，不允许删除信息，必须全部给出。"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"以下是来自图纸与铭牌识别的字段信息，请将内容按如下五类进行整理：\n\n{text_to_summarize}\n\n"
                        "1. **零件几何参数**：包括尺寸、直径、长度、孔、键槽、中心距、数量、角度等所有物理形状参数；\n"
                        "2. **精度与公差要求**：包括各种公差数值、形位误差、配合等级、精度等级、允许偏差等；\n"
                        "3. **表面与加工要求**：包括倒角、表面粗糙度、光洁度、加工工艺、未注要求等；\n"
                        "4. **材料与热处理**：包括材料名称、性能标准、热处理方法、淬火深度、硬度、组织要求等；\n"
                        "5. **其他**：凡不属于以上四类的信息（如检测标准、探伤方法、执行规范、标注符号说明等）统一放入该类；\n\n"
                        "请严格按照以下格式输出：两层 JSON，第一层为上述五大分类，第二层为“字段名: 字段值”键值对，绝对不允许任何嵌套（即字段值不能是对象/数组，不能出现大括号{}或中括号[]）。\n\n"
                        "“字段名: 字段值”对，只允许一对一对出现；不添加多余解释或非 JSON 内容，不可再增加json层数。\n\n"
                        "对同类信息用“位置/属性+核心名称”的格式细化，例如“主视图倒角”“齿面表面粗糙度”“中心孔直径”，避免用“倒角”“直径”等模糊名称导致需要嵌套区分。"
                        "错误示例：\n"
                        "{\n"
                        "  \"零件几何参数\": {\"倒角\": {\"主视图\": \"2*45\"，\"俯视图\": \"2*45\"}, \"总长度\": \"780±0.5\", \"孔径\": \"Φ110\"},\n"
                        "}"
                        "正确示例：\n"
                        "{\n"
                        "  \"零件几何参数\": {\"零件名称\": \"齿轮\", \"总长度\": \"780±0.5\", \"孔径\": \"Φ110\"},\n"
                        "  \"精度与公差要求\": {\"齿距偏差\": \"±0.0085\"},\n"
                        "  \"表面与加工要求\": {\"倒角\": \"C1\", \"表面粗糙度\": \"Ra1.6\"},\n"
                        "  \"材料与热处理\": {\"材料\": \"20CrMnMo\", \"硬度\": \"59~62HRC\"},\n"
                        "  \"其他\": {\"执行标准\": \"GB/T 3077-1999\"}\n"
                        "}"
                    )
                }
            ],
            stream=False
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"总结失败: {str(e)}"

# ========= 调用RAGFlow知识库获取工艺 =========
def call_ragflow_summary_api(summary_raw: str,
                              chat_id: str,
                              api_key: str,
                              ragflow_address: str,
                              model: str = "model",
                              stream: bool = False,
                              reference: bool = False):
    client = OpenAI(
        api_key=api_key,
        base_url=f"http://{ragflow_address}/api/v1/chats_openai/{chat_id}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个懂机械加工的智能助手，请根据用户提供的结构化零件信息进行工艺分析，结合你的知识库，"
                "识别该零件在加工制造过程中可能涉及的主要工序（如粗加工、热处理、精加工、检测等），"
                "并按如下格式输出一个**一层结构的 JSON 对象**：\n\n"
                "{\n"
                "  \"工序编码\": \"工序名称\",\n"
                "  \"工序编码\": \"工序名称\"\n"
                "}\n\n"
                "只返回 JSON，不要添加多余解释。确保编码唯一、名称准确。如信息不足，请合理推理填充。"
            )
        },
        {
            "role": "user",
            "content": summary_raw.strip()
        }
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
        extra_body={"reference": reference}
    )

    if stream:
        final_content = ""
        reference_data = None
        for chunk in completion:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content"):
                final_content += delta.content
            if reference and chunk.choices[0].finish_reason == "stop":
                reference_data = delta.reference
        return final_content.strip()
    else:
        return completion.choices[0].message.content.strip()


def load_process_data(excel_file_path):
    """
    加载Excel中的工艺数据，包括工艺编码、名称、计费方式和成本

    参数:
        excel_file_path: Excel文件的路径

    返回:
        tuple: (工艺数据字典, 错误信息)
              - 成功时：(process_data, None)
              - 失败时：(None, error_message)
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(EXCEL_FILE_PATH):
            return None, f"Excel文件不存在: {EXCEL_FILE_PATH}"

        # 读取Excel文件，使用第一行作为表头
        df = pd.read_excel(EXCEL_FILE_PATH, header=0)

        # 验证必要列是否存在（新增“工艺编码”列的检查）
        required_columns = ["工艺编码", "工艺名称", "计费方式", "工艺成本"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None, f"Excel文件缺少必要列: {', '.join(missing_columns)}"

        # 构建工艺数据字典（以“工艺编码”为键）
        process_data = {}
        for _, row in df.iterrows():
            # 提取并清洗数据
            process_code = str(row["工艺编码"]).strip()  # 工艺编码（作为键）
            process_name = str(row["工艺名称"]).strip()
            billing_method = str(row["计费方式"]).strip()

            # 验证工艺成本是否为有效数字
            try:
                cost = float(row["工艺成本"])
                # 存储数据（包含编码、名称、计费方式、成本）
                process_data[process_code] = {
                    "工艺名称": process_name,
                    "计费方式": billing_method,
                    "工艺成本": cost
                }
            except ValueError:
                return None, f"工艺编码'{process_code}'的成本不是有效的数字"

        return process_data, None

    except Exception as e:
        return None, f"加载工艺数据时出错: {str(e)}"

# ========= 接口：上传图像并返回识别果 =========
@app.route('/segment_and_recognize', methods=['POST'])
def segment_and_recognize():
    if 'image' not in request.files:
        return jsonify({'error': '未检测到图像文件'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    # 创建临时目录保存文件
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, file.filename)
        file.save(input_path)

        # 时间戳输出目录
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(temp_dir, f"output_{time_str}")
        os.makedirs(output_dir, exist_ok=True)

        # 图像分割处理
        result_images = process_image(input_path, output_dir)

        if len(result_images) < 2:
            return jsonify({'error': '图像中未识别出两个有效区域'}), 200

        # OpenAI 识别处理
        base64_img1 = encode_image_base64(result_images[0])
        base64_img2 = encode_image_base64(result_images[1])

        try:
            client = OpenAI(
                api_key="",  # 建议保存在环境变量中
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            completion = client.chat.completions.create(
                model="qwen2.5-vl-72b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img1}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img2}"}},
                            {"type": "text", "text":
                             "请识别以下两张图片中的关键信息，并以 “字段名: 字段值” 的形式，所有内容请使用中文表达："
                             "第一张图片：为零件铭牌照片，表格形式居多，请理解表格的排列后，请尽可能准确提取所有字段，并输出为“字段名: 字段值”的格式。"
                             "结合工业零件制造常识，判断字段的可能归属"
                             "第二张图片：为零件图纸"
                             "识别所有出现的零件图形,提取其相关的尺寸参数（如：直径、半径、长度、宽度、角度、公差等）,若存在多个细节图，请逐一列出，并注明放大视图对应的部位,若有“锻造要求”或其他技术说明，也请提取出来"
                             "所有数据以 “字段名: 字段值” 形式表示，例如："
                             "模数（平均）: 8,齿数: 23,直径: 45mm,锻造要求: 热处理至HRC58-62"}
                        ]
                    }
                ]
            )

            # 原始识别结果
            raw_result = completion.choices[0].message.content

            # 调用 DeepSeek 模型进行总结
            summary_raw = summarize_with_deepseek(raw_result)

            # 提取纯 JSON 内容（去掉 markdown 包裹的 ```json 和 ```）
            cleaned = re.sub(r"^```json\s*|\s*```$", "", summary_raw.strip())

            # 尝试转为 JSON 对象
            try:
                summary_json = json.loads(cleaned)
            except Exception as e:
                summary_json = {"解析失败": f"结构化总结转换失败，请人工查看。错误：{str(e)}", "原始": summary_raw}


            # 返回两个部分
            return jsonify({
                '总结': summary_json
            })

        except Exception as e:
            return jsonify({'error': f'调用识别服务失败: {str(e)}'}), 500

# ========= 接口：从工艺获得零件价格 =========
@app.route('/get_process_costs', methods=['POST'])
def get_process_costs():
    """处理工艺成本查询请求，支持两种JSON格式"""
    # 加载工艺数据
    process_data, error = load_process_data(EXCEL_FILE_PATH)
    if error:
        return jsonify({"error": error}), 500

    try:
        # 获取请求数据
        request_data = request.get_json()
        if not request_data:
            return jsonify({"error": "请求数据不能为空"}), 400

        # 解析工艺数据（支持两种格式）
        if "工艺" in request_data and isinstance(request_data["工艺"], dict):
            processes = request_data["工艺"]
        else:
            processes = request_data
            if not isinstance(processes, dict):
                return jsonify({"error": "输入格式错误，应为字典或包含'工艺'键的字典"}), 400

        # 处理查询结果
        result = {"工艺": {}}
        for index, (process_code, process_info) in enumerate(processes.items(), start=1):
            # 生成两位数序号（如01、02）
            seq = f"{index:02d}"

            # 从工艺数据中获取完整信息
            if process_code in process_data:
                full_info = process_data[process_code]
                result["工艺"][seq] = {
                    "工艺编码": process_code,
                    "工艺名称": full_info["工艺名称"],
                    "计费方式": full_info["计费方式"],
                    "工艺成本": full_info["工艺成本"]
                }
            else:
                # 未找到对应工艺时的处理
                result["工艺"][seq] = {
                    "工艺编码": process_code,
                    "工艺名称": "未找到",
                    "计费方式": "未找到",
                    "工艺成本": "未找到"
                }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"处理请求时出错: {str(e)}"}), 500

# ========= 入口 =========
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
