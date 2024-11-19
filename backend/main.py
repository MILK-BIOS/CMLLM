import fastapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = fastapi.FastAPI()

# 定义请求体的数据模型
class Message(BaseModel):
    message: str

@app.get("/")
def read_root():
    return JSONResponse(content={"reply": "Hello, world!"})

@app.get("/hello")
def hello():
    return JSONResponse(content={"reply": "你好我是中医体质测评助手，请尽可能准确地回答一下问题，大约需要10分钟左右，让我们开始吧！"})

@app.post("/chat")
def chat(request: Message):
    print(request)
    return JSONResponse(content={"reply": f"Hello, {request.message}!"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9495)