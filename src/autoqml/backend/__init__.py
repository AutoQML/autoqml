from dataclasses import dataclass
from typing import Dict, Any

# TODO: Remove?

@dataclass
class Provider:
    name: str
    api_token: str


class IBM(Provider):

    def __init__(self, api_token: str):
        super().__init__('IBM', api_token)


@dataclass
class Backend:
    provider: Provider
    name: str
    executor_kwargs: Dict[str, Any]
