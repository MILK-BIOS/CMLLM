import langchain
from langchain_core.prompts import PromptTemplate

class SoulutionProposer:
    def __init__(
        self, 
        llm, 
        template: str=None
    ):
        self.llm = llm
        if template:
            self.template = template
        else:
            self.template = """
                You are an medical advisor of traditional Chinese medicine constitution.

                Now there is a human's constitution classification result. Fisrt show the result and then give your advice.

                The result is as follows:
                {result}

                And propensity result is as follows:
                {pretend_result}
                
                Please give a comprehensive advice in Chinese and avoid using word like patient.
            """
        self.prompt = PromptTemplate.from_template(self.template)
        self.chain = self.prompt | self.llm

    def invoke(self, input):
        return self.chain.invoke(input)