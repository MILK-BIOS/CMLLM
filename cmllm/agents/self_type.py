import langchain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.tools.render import render_text_description
from langchain_core.prompts import PromptTemplate
from ..utils import Choice, MyPrintHandler, Score
from pydantic import ValidationError


class TypeClassifier:
    def __init__(self, llm, sheet, tools, template=None):
        self.llm = llm
        self.sheet = sheet
        if template:
            self.template = template
        else:
            self.template = """
                            You are now an doctor of traditional Chinese medicine constitution classification.
                            These constitutions include nine kinds, namely, 平和体质、阳虚体质、阴虚体质、气虚体质、痰湿体质、湿热体质、淤血体质、气郁体质、特禀体质. Except for the 平和体质, all belong to the 偏颇体质.
                            Now there's a physical fitness scale, and you score them based on their answers.

                            Scoring method and judging criteria:
                            There are 5 levels of answers under each question, and the score value of 1 to 5 points is given from the tendency of none to some (the items marked with * are the reverse scoring items), and the original score of each type is selected in a single choice way, and then the simple summation method is used.

                            Raw score = the sum of the branches of each question.
                            After every question, you can use the tool to record and calculate conversion scores from raw score.
                            Conversion score =[(original score - number of entries)/ Number of entries ×4].

                            constitution_standard:
                            {constitution_standard}

                            Tools as follows:
                            {tools}

                        """
        self.prompt = PromptTemplate.from_template(self.template).partial(
            constitution_standard = self.sheet,
            tools = render_text_description(tools))
        self.chain = self.prompt | self.llm

    def invoke(self, input):
        return self.chain.invoke(input)

class Valuator:
    def __init__(self, llm, sheet, tools, template=None):
        self.llm = llm
        self.sheet = sheet
        if template:
            self.template = template
        else:
            self.template = """
                            You are now a valuator of traditional Chinese medicine constitution classification question. 
                            Your mission is to rate the user's input.

                            Scoring method and judging criteria:
                            There are 5 levels of answers under each question, and the score value of 1 to 5 points is given from the tendency of none to some (the items marked with * are the reverse scoring items). Please give your rating according to the overall meaning.

                            Evaluate_standard:
                            {evaluate_standard}
                            
                            Especially Attention! If the question end with *, please reverse the standard!

                            After every question, you should use the tool to record the score.

                            Tools as follows:
                            {tools}

                            The question is:
                            {question}

                            The user's answer needed to score as follows:
                            {answer}

                            Finally, print the tools chose above exactly and just pass the score number you rate
                            {format_instructions}
                        """
        self.output_parser = PydanticOutputParser(pydantic_object=Choice)
        self.tools = tools
        self.prompt = PromptTemplate.from_template(self.template).partial(
            evaluate_standard = self.sheet,
            tools = render_text_description(tools),
            format_instructions = self.__chinese_friendly(
                    self.output_parser.get_format_instructions(),
                )
            )
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.verbose_printer = MyPrintHandler()

    def invoke(self, input):
        choice = self.chain.invoke(input)
        start = choice.find("{")  # format json
        end = choice.rfind("}")
        if start != -1 and end != -1 and start < end:
            choice = choice[start:end + 1]  # 切片操作，包含最后一个 '}'
            choice = self.output_parser.parse(choice)
        else:
            return None  # 如果没有找到匹配的 '{' 和 '}'，返回 None
        observation = self.__exec_action(choice)
        return observation

    def __exec_action(self, choice: Choice):
        observation = "No relative tool found"
        args = choice.model_dump()
        del args['name']
        for tool in self.tools:
            if tool.name == choice.name:
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
        