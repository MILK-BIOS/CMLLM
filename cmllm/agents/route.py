from .task_dispatcher import TaskDispatcher

def route(info):
    if "知识问答" in info["topic"].lower():
        return knowledge_chain.invoke(info)
    elif "数学计算" in info["topic"].lower():
        return math_chain.invoke(info)
    elif "生活指南" in info["topic"].lower():
        return life_chain.invoke(info)
    else:
        return other_chain.invoke(info)
