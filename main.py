import fastapi
from langchain.tools import Tool
from langchain_community.llms.moonshot import Moonshot
from fastapi import HTTPException, Depends, BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from cmllm import Score, Valuator, SoulutionProposer, type_judge
import numpy as np
import matplotlib.pyplot as plt
import uuid
import os
import cv2
import base64
import time
import asyncio



async def periodic_cleanup():
    while True:
        cleanup_states()
        await asyncio.sleep(1)  # Run cleanup every 30 seconds

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting periodic cleanup task...")
    cleanup_task = asyncio.create_task(periodic_cleanup())  # Start background task
    yield
    print("Stopping periodic cleanup task...")
    cleanup_task.cancel()  # Stop the background task on shutdown
    try:
        await cleanup_task
    except asyncio.CancelledError:
        print("Periodic cleanup task stopped.")
 
app = fastapi.FastAPI(lifespan=lifespan)
# 全局状态存储，每个客户端有独立的状态
states = {}
standard = {
    "没有(根本不)": 1,
    "不会": 1,
    "很少(有一点)": 2,
    "偶尔": 2,
    "有时(有些)": 3,
    "还可以": 3,
    "经常(相当)": 4,
    "总是(非常)": 5
}
# load questions
folder_path = './questions'

cat = {0: '平和体质', 
       1: '阴虚体质', 
       2: '痰湿体质', 
       3: '气虚体质', 
       4: '特禀体质', 
       5: '阳虚体质', 
       6: '湿热体质', 
       7: '淤血体质', 
       8: '气郁体质'}

questions = {}

for filename in os.listdir(folder_path):
    # 构建每个文件的完整路径
    file_path = os.path.join(folder_path, filename)
    catagory = os.path.splitext(filename)[0]
    # 检查是否为文件
    if os.path.isfile(file_path):
        print(f"Processing file: {filename}")
        # 打开并逐行读取文件中的问题
        questions[catagory] = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                question = line.strip()  # 去掉行末尾的空白字符
                if question:
                    questions[catagory].append(question)

STATE_EXPIRATION_TIME = 1800

def cleanup_states():
    # 定期清理过期状态
    current_time = time.time()
    to_delete = [client_id for client_id, state in states.items()
                 if current_time - state["last_accessed"] > STATE_EXPIRATION_TIME]
    for client_id in to_delete:
        del states[client_id]
        img_path = f'./result/img/{client_id}.png'
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"File {img_path} deleted successfully.")
        print(f"State for client {client_id} deleted due to inactivity.")

# 生成或获取客户端 ID
def get_client_id(client_id: str = None):
    print(client_id)
    if not client_id:
        client_id = str(uuid.uuid4())  # 如果未提供 client_id，则生成新的 UUID
    if client_id not in states:
        states[client_id] = None  # 初始化为 None，等待初始化
    return client_id

def initialize_state():
    score = Score()
    score_tools = [
    Tool.from_function(
        func=score.calculate_final,
        name="Calculate conversion scores",
        description="useful for when you need to calculate conversion scores from raw score"
        # coroutine= ... <- you can specify an async method if desired as well
    ),
    Tool.from_function(
        func=score.update,
        name="Update score after questions",
        description="useful for when you need to record score of answer"
        # coroutine= ... <- you can specify an async method if desired as well
    )
    ]
    os.environ["MOONSHOT_API_KEY"] = "sk-jp2VoAvlcR3QS8azmXmFIRqxLA4nEVF4o48j8EkJilkm3DfV"
    llm = Moonshot(temperature=0.1)
    va = Valuator(llm, standard, score_tools)
    ad = SoulutionProposer(llm)
    return {
        "score": score,
        "valuator": va,
        "current_catagory_index": 0,
        "current_question_index": 0,
        "advisor": ad,
        "last_accessed": time.time()
    }

def final_show(final, client_id):
    plt.rcParams['font.sans-serif'] = ["SimHei"]  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
    # 获取标签和数值
    labels = list(final.keys())
    values = list(final.values())
    num_vars = len(labels)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # 闭合图形
    values += values[:1]
    angles += angles[:1]

    # 初始化雷达图
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

    # 绘制雷达图
    ax.fill(angles, values, color='skyblue', alpha=0.4)
    ax.plot(angles, values, color='blue', linewidth=2)

    # 添加标签
    ax.set_ylim(0, 70) 
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(pad=20)

    # 显示图形
    fig.savefig(f'./result/img/{client_id}.png', bbox_inches='tight')
    plt.close(fig)

# 定义请求体的数据模型
class Message(BaseModel):
    message: str

@app.get("/")
def read_root():
    return JSONResponse(content={"reply": "Hello, world!"})

@app.post("/hello")
def hello(client_id: str = Depends(get_client_id)):

    states[client_id] = initialize_state()
    return JSONResponse(content={
        "reply": "你好我是中医体质测评助手，请尽可能准确地回答一下问题，大约需要10分钟左右，让我们开始吧！",
        "client_id": client_id,})

@app.post("/start")
def start(client_id: str):
    print(client_id)
    if client_id not in states or states[client_id] is None:
        raise HTTPException(status_code=400, detail="Client not initialized. Please call /hello/ first.")
    state = states[client_id]
    current_catagory_index = state["current_catagory_index"]
    current_question_index = state["current_question_index"]
    state["current_question_index"] += 1
    return JSONResponse(content={
        "reply": questions[cat[current_catagory_index]][current_question_index],
        "client_id": client_id,})

@app.post("/chat")
def chat(request: Message, client_id: str = Depends(get_client_id)):
    if client_id not in states or states[client_id] is None:
        raise HTTPException(status_code=400, detail="Client not initialized. Please call /hello/ first.")

    # Retrieve the current state
    state = states[client_id]
    states["last_accessed"] = time.time()
    score_client = state["score"]
    va = state["valuator"]
    current_catagory_index = state["current_catagory_index"]
    current_question_index = state["current_question_index"]
    que = questions[cat[current_catagory_index]][current_question_index]

    # Process the answer
    ans = request.message
    va.invoke({'question': que, 'answer': ans})

    # Check if we have reached the end of the current category
    if current_question_index + 1 >= len(questions[cat[current_catagory_index]]):
        score_client.record(cat[current_catagory_index])
        state["current_catagory_index"] += 1
        state["current_question_index"] = 0

        # Check if we have reached the end of all categories
        if state["current_catagory_index"] >= len(cat):
            return JSONResponse(content={
                "reply": "测试完成报告生成中...",
                "finish": True,
            })

        que_next = questions[cat[state["current_catagory_index"]]][0]
    else:
        state["current_question_index"] += 1
        que_next = questions[cat[current_catagory_index]][current_question_index + 1]

    return JSONResponse(content={
        "reply": f"好的，下一个问题: {que_next}",
        "finish": False,
    })

@app.post("/result")
def result(client_id: str = Depends(get_client_id)):
    # Retrieve the current state
    if client_id not in states or states[client_id] is None:
        raise HTTPException(status_code=400, detail="Client not initialized. Please call /hello/ first.")
    state = states[client_id]
    states["last_accessed"] = time.time()
    ad = state["advisor"]
    score_client = state["score"]
    final = score_client.final

    # Calculate final result
    final_show(final, client_id)
    file_path = f'./result/img/{client_id}.png'
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    type_result = type_judge(final)
    final_type = [key for key, value in type_result.items() if value == 2]
    tend_type = [key for key, value in type_result.items() if value == 1]
    text_result = ad.invoke({"result": final_type, "pretend_result": tend_type})
    return JSONResponse(content={
        "image": encoded_string,
        "text": text_result,
    })

# only one question for test
@app.post("/chat_test")
def chat(request: Message, client_id: str = Depends(get_client_id)):
    if client_id not in states or states[client_id] is None:
        raise HTTPException(status_code=400, detail="Client not initialized. Please call /hello/ first.")

    # Retrieve the current state
    state = states[client_id]
    score_client = state["score"]
    va = state["valuator"]
    current_catagory_index = state["current_catagory_index"]
    current_question_index = state["current_question_index"]
    que = questions[cat[current_catagory_index]][current_question_index]

    if state["current_question_index"] >= 1:
        score_client.nums = 60
        score_client.record(cat[current_catagory_index])
        return JSONResponse(content={
            "reply": "测试完成报告生成中...",
            "finish": True,
        }) # TEST TEST TEST

    # Process the answer
    ans = request.message
    va.invoke({'question': que, 'answer': ans})

    # Check if we have reached the end of the current category
    if current_question_index + 1 >= len(questions[cat[current_catagory_index]]):
        score_client.record(cat[current_catagory_index])
        state["current_catagory_index"] += 1
        state["current_question_index"] = 0

        # Check if we have reached the end of all categories
        if state["current_catagory_index"] >= len(cat):
            return JSONResponse(content={
                "reply": "测试完成报告生成中...",
                "finish": True,
            })

        que_next = questions[cat[state["current_catagory_index"]]][0]
    else:
        state["current_question_index"] += 1
        que_next = questions[cat[current_catagory_index]][current_question_index + 1]

    return JSONResponse(content={
        "reply": f"好的，下一个问题: {que_next}",
        "finish": False,
    })

# result according to the chat test 
@app.post("/result_test")
def result(client_id: str = Depends(get_client_id)):
    # Retrieve the current state
    if client_id not in states or states[client_id] is None:
        raise HTTPException(status_code=400, detail="Client not initialized. Please call /hello/ first.")
    state = states[client_id]
    ad = state["advisor"]
    score_client = state["score"]
    final = score_client.final
    final = {'平和体质': 46.875, '阴虚体质': 18.75, '痰湿体质': 37.5, '气虚体质': 28.125, '特禀体质': 32.142857142857146, '阳虚体质': 32.142857142857146, '湿热体质': 42.857142857142854, '淤血体质': 3.571428571428571, '气郁体质': 10.714285714285714}
    print(final) # TEST TEST TEST
    
    # Calculate final result
    final_show(final, client_id)
    file_path = f'./result/img/{client_id}.png'
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    type_result = type_judge(final)
    final_type = max(type_result, key=type_result.get)
    tend_type = [key for key, value in type_result.items() if value == 1]
    text_result = ad.invoke({"result": final_type, "pretend_result": tend_type})
    return JSONResponse(content={
        "image": encoded_string,
        "text": text_result,
    })
    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9495)