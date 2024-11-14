import os
from typing import List
from langchain_core .runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from langchain.chains import GraphCypherQAChain
from langchain_community.llms.moonshot import Moonshot
from langchain_community.graphs import Neo4jGraph
from cmllm import TaskDispatcher

os.environ["NEO4J_URI"] = "neo4j://localhost:7774"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "neo4jneo4j"
os.environ["MOONSHOT_API_KEY"] = "sk-jp2VoAvlcR3QS8azmXmFIRqxLA4nEVF4o48j8EkJilkm3DfV"


graph = Neo4jGraph(url=os.environ["NEO4J_URI"],
                    username=os.environ["NEO4J_USERNAME"], 
                    password=os.environ["NEO4J_PASSWORD"],           
                    database='tcm',
                    enhanced_schema=True)

kg_retrieve_chain = GraphCypherQAChain.from_llm(
    Moonshot(), graph=graph, verbose=True, allow_dangerous_requests=True
)

knowledge_retrieve_tool = StructuredTool.from_function(
    func=kg_retrieve_chain.invoke,
    name="查询病症",
    description="查询病症病因",
)
# knowledge_retrieve_tool.run({"input":{'query': '口干舌燥是什么病', 'tags': None}})



finish_placeholder = StructuredTool.from_function(
    func=lambda: None,
    name="FINISH",
    description="用于表示任务完成的占位符工具"
)

tools = [knowledge_retrieve_tool, finish_placeholder]

llm = Moonshot(
    temperature=0.2,
    timeout=30
    ) 
my_agent = TaskDispatcher(llm=llm, tools=tools)
task = "腹痛可能有哪些病因"
reply = my_agent.run(task)
print(reply)