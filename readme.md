# 这是一个图纸核价项目

通过unet进行图纸的分割，得到图形区域和BOM清单，分别对这两张分割后的图形进行视觉大模型的识别，得到零件的参数，然后再利用大模型对两张图的参数进行整合。将整合后的参数利用RAG，得到该零件的制作工艺，然后利用一个excel进行核价计算。

# 1.创建环境

为防止环境冲突，该项目需要需要两个环境：python3.8与python3.12。

#### 创建 Python 3.8 环境（命名为`unet`）

```bash
sudo apt update
sudo apt install python3.8 python3.8-venv && python3.8 -m venv unet
```

#### 创建 Python 3.12 环境（命名为`ragflow_agent`）

```bash
sudo apt install python3.12 python3.12-venv && python3.12 -m venv ragflow_agent
```

# 2.安装依赖

激活`unet`环境后，执行以下命令安装所有依赖：

```bash
# 先安装PyTorch（带CUDA 11.3）
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 再安装其他依赖（通过requirements.txt）
pip install -r requirements.txt
```

激活`ragflow_agent`环境后，执行：

```bash
pip install -r requirements.txt
```