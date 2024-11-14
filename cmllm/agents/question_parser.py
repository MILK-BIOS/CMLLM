import langchain


class QuestionParser:
    def __init__(
        self, 
        llm, 
        graph, 
        template: str=None
    ):
        self.llm = llm
        self.graph = graph
        if template:
            self.template = template
        else:
            self.template = """
                You are an expert of Medical Science. Now you get a question from patient.
                Transform the patient's condition into the specific attributes that the Knowledge graph node has.
                Especially compare and transform the symptoms.
                Node type and attributes as follows：
                ---
                {schema}
                ---
                question as follows：
                ---
                {question}
                ---
                Now please write the query：
            """