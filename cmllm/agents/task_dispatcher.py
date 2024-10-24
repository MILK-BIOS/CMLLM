import langchain
from typing import Optional, Tuple
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from ..utils import Action

class TaskDispatcher:
    def __init__(
        self, 
        llm, 
        tools, 
        prompt: str='', 
        final_prompt: str='', 
        max_thought_steps: Optional[int]=10, 
        template: str=None
    ):
        self.llm = llm
        self.tools = tools
        self.final_prompt = PromptTemplate.from_template(final_prompt)
        self.max_thought_steps = max_thought_steps # 最多思考步数，避免死循环
        self.output_parser = PydanticOutputParser(pydantic_object=Action)

        if template:
            self.template = template
        else:
            self.template = """
                你是强大的AI中医助手，可以使用工具与指令查询并相关资料并给出相应解决方案

                你的任务是:
                {task_description}

                你可以使用以下工具或指令，它们又称为动作或actions:
                {tools}

                当前的任务执行记录:
                {memory}

                按照以下格式输出：

                任务：你收到的需要执行的任务
                思考: 观察你的任务和执行记录，并思考你下一步应该采取的行动
                然后，根据以下格式说明，输出你选择执行的动作/工具:
                {format_instructions}
                """

        self.output_prompt = """
                        你的任务是:
                        {task_description}

                        以下是你的思考过程和使用工具与外部资源交互的结果。
                        {memory}

                        你已经完成任务。
                        现在请根据上述结果简要总结出你的最终答案。
                        直接给出答案。不用再解释或分析你的思考过程。
                        """

    def run(self, task_description: str=''):
        """Main progress of Chinese Medicine assistant

        Combine all agents and tools to generate the answer expected.

        Args: 
          task_description: Pass the message of your question

        Result:
          LLM's answer according to the KG and reference
        """
        thought_step_count = 0
        agent_mem = ConversationTokenBufferMemory(llm=self.llm)

        while thought_step_count < self.max_thought_steps:
            print(f">>>>Round: {thought_step_count}<<<<")
            action, response = self.__step(
                task_description=task_description,
                memory=agent_mem
            )

            # 如果是结束指令，执行最后一步
            if action.name == "FINISH":
                break

            # 执行动作
            observation = self.__exec_action(action)
            print(f"----\nObservation:\n{observation}")

            # 更新记忆
            self.__update_memory(agent_memory, response, observation)

            thought_step_count += 1

        if thought_step_count >= self.max_thought_steps:
            # 如果思考步数达到上限，返回错误信息
            reply = "抱歉，我没能完成您的任务。"
        else:
            # 否则，执行最后一步
            final_chain = self.final_prompt | self.llm | StrOutputParser()
            reply = final_chain.invoke({
                "task_description": task_description,
                "memory": agent_memory
            })

        return reply

    def __step(self, task_description, memory) -> Tuple[Action, str]:
        """Step one loop"""
        response = ""
        for s in self.llm_chain.stream({
            "task_description": task_description,
            "memory": memory
        }, config={
            "callbacks": [
                self.verbose_printer
            ]
        }):
            response += s

        action = self.output_parser.parse(response)
        return action, response

    def __exec_action(self, action: Action) -> str:
        observation = "没有找到工具"
        for tool in self.tools:
            if tool.name == action.name:
                try:
                    # 执行工具
                    observation = tool.run(action.args)
                except ValidationError as e:
                    # 工具的入参异常
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    # 工具执行异常
                    observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"

        return observation

    @staticmethod
    def __update_memory(agent_memory, response, observation):
        agent_memory.save_context(
            {"input": response},
            {"output": "\n返回结果:\n" + str(observation)}
        )

    @staticmethod
    def __chinese_friendly(string) -> str:
        lines = string.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('{') and line.endswith('}'):
                try:
                    lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
                except:
                    pass
        return '\n'.join(lines)

    
