import langchain
import json
from typing import Optional, Tuple
from langchain_core.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import ValidationError
from ..utils import Action, MyPrintHandler

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
        self.max_thought_steps = max_thought_steps # 最多思考步数，避免死循环
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        

        if template:
            self.template = template
        else:
            self.template = """
                你是强大的AI中医助手的任务调度者，可以使用给定的工具或解决者查询并相关资料并给出相应解决方案

                你的任务是:
                {task_description}

                你可以使用以下工具或指令，它们又称为解决者:
                {tools}

                当前的任务执行记录:
                {memory}

                按照以下格式输出：

                任务：你收到的需要执行的任务，并转发给相应的解决者
                思考: 观察你的任务和执行记录，并思考你下一步应该采取的行动
                然后，根据以下格式说明，输出你选择执行的一个解决者/tool，你需要在args中直接使用字符串给解决者prompt:
                {format_instructions}
                """
        
        if final_prompt:
            self.final_prompt = PromptTemplate.from_template(final_prompt)
        else:
            output_prompt = """
                        你的任务是:
                        {task_description}

                        以下是你的思考过程和使用工具与外部资源交互的结果。
                        {memory}

                        你已经完成任务。
                        现在请根据上述结果简要总结出你的最终答案。
                        直接给出答案。不用再解释或分析你的思考过程。
                        """
            self.final_prompt = PromptTemplate.from_template(output_prompt)
        def __init_prompt(prompt):
            return PromptTemplate.from_template(prompt).partial(
                tools=render_text_description(self.tools),
                format_instructions=self.__chinese_friendly(
                    self.output_parser.get_format_instructions(),
                )
            )
        self.prompt = __init_prompt(self.template)
        self.llm_chain = self.prompt | self.llm | StrOutputParser()
        self.verbose_printer = MyPrintHandler()



    def run(self, task_description: str=''):
        """Main progress of Chinese Medicine assistant

        Combine all agents and tools to generate the answer expected.

        Args: 
          task_description: Pass the message of your question

        Result:
          LLM's answer according to the KG and reference
        """
        thought_step_count = 0
        agent_memory = ConversationTokenBufferMemory(llm=self.llm)

        while thought_step_count < self.max_thought_steps:
            print(f">>>>Round: {thought_step_count}<<<<")
            action, response = self.__step(
                task_description=task_description,
                memory=agent_memory
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
            print('-------------------------')

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
        json_arg = response[response.find('{'):]
        action = self.output_parser.parse(json_arg)
        return action, response

    def __exec_action(self, action: Action) -> str:
        observation = "没有找到工具"
        args = {"input": action.prompt.dict()}
        for tool in self.tools:
            if tool.name == action.name:
                try:
                    # 执行工具
                    observation = tool.run(args)
                except ValidationError as e:
                    # 工具的入参异常
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {args}"
                    )
                except Exception as e:
                    # 工具执行异常
                    observation = f"Error: {str(e)}, {type(e).__name__}, args: {args}"

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
                    print("Success")
                except:
                    pass
        return '\n'.join(lines)

    
