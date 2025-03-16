from typing import Iterator, List, Union
import owui_pipelines as owui



class Pipeline( owui.api.Pipeline ):
    def __init__(self):
        super().__init__()
        self.name = "Legal Cases Agent"

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[Iterator[str], str]:
        return super().pipe(user_message, model_id, messages, body)
        
                