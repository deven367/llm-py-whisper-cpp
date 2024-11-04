import llm

@llm.hookimpl
def register_models(register):
    register(PyWhisper())

class PyWhisper(llm.Model):
    model_id = "pywhisper"

    def execute(self, prompt, stream, response, conversation):
        return ["hello world"]
