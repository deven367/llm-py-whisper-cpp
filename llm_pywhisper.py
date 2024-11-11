import llm
AVAILABLE_MODELS = [
    "base",
    "base-q5_1",
    "base.en",
    "base.en-q5_1",
    "large-v1",
    "large-v2",
    "large-v2-q5_0",
    "large-v3",
    "large-v3-q5_0",
    "medium",
    "medium-q5_0",
    "medium.en",
    "medium.en-q5_0",
    "small",
    "small-q5_1",
    "small.en",
    "small.en-q5_1",
    "tiny",
    "tiny-q5_1",
    "tiny.en",
    "tiny.en-q5_1",
    "tiny.en-q8_0",
]


@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("audio_file", type=str)
    @click.option(
        "--model",
        type=click.Choice(AVAILABLE_MODELS),
        default="base",
        help="Whisper model to use",
    )
    def pywhisper(audio_file, model:str):
        model = Model(model, n_threads=6)
        segments = model.transcribe(audio_file, language='auto', single_segment=True, print_timestamps=True, print_realtime=True)
        for segment in segments:
            click.echo(segment.text)


    # register(PyWhisper())

class PyWhisper(llm.Model):
    model_id = "pywhisper"

    def execute(self, prompt, stream, response, conversation):
        return ["hello world"]
