### 项目介绍
1. api在官方原版的基础上，通过SSE协议实现api流式传输。注：*SSE（Server-sent Events）是 WebSocket 的一种轻量代替方案，是一种服务器端到客户端（浏览器）的单向流式消息推送协议，在AI 生成对话等场景下较为常见。 Web 函数目前已经支持通过SSE 协议在客户端和函数运行的服务端间建立连接。*
2. 将官方原版web_demo中的预测函数替换为requests请求，这样可以在开启api的同时支持web页面。


### 准备工作
1. 提前安装好pytorch（建议GPU版），然后安装环境
```bash
pip install "fastapi[all]"
pip install -r requirements.txt
```
2. 去huggingface下载ChatGLM-6b模型到本项目，并且将`chatglm-6b`重命名为`chatglm_6b`，可以参考下面的命令：
```bash
git lfs install
git clone --depth=1 https://huggingface.co/THUDM/chatglm-6b
mv chatglm-6b chatglm_6b
```
3. 修改一下代码chatglm_6b目录下面的`modeling_chatglm.py`文件(本项目里面提供了这个文件，不过可能不是最新版，仅供参考)，搜索`stream_chat`函数，在for循环后面增加一个`yield None None`表示本次对话结束。
- 修改前
```python
for outputs in self.stream_generate(**inputs, **gen_kwargs):
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs)
    response = self.process_response(response)
    new_history = history + [(query, response)]
    yield response, new_history
```
- 修改后
```python
for outputs in self.stream_generate(**inputs, **gen_kwargs):
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(outputs)
    response = self.process_response(response)
    new_history = history + [(query, response)]
    yield response, new_history
yield None, None
```

### 正式运行
1. 启动api(必须)
```bash
python3 api.py
```

2. 启动web_demo，然后浏览器访问：ip:7860即可。例如:http://localhost:7860
```bash
python3 web_demo.py
```

3. 启动cli版客户端,可以和官方cli版一样在终端聊天，不过这个是用SSE协议在请求api。
```bash
python3 client.py
```