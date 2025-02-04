from typing import Union, Any, Dict

from autoqml.constants import TrialId
from autoqml.util.singleton import Singleton


@Singleton
class ConfigContext:

    def __init__(self):
        self.store: Dict[TrialId, Dict[str, Any]] = dict()

    def set_config(self, id: TrialId, config: Dict = None, key: str = None, value: Any = None) -> None:
        if config is None:
            if key is None:
                raise ValueError('Config and key/value pair are both None')
            config = {key: value}
        else:
            if key is not None or value is not None:
                raise ValueError(f'Provide either config or key/value pair, not both')

        if id not in self.store:
            self.store[id] = dict()
        self.store[id].update(config)

    def get_config(self, id: TrialId, key: str = None, default: Any = None) -> Union[Any, Dict[str, Any]]:
        if key is None:
            return self.store[id]
        else:
            return self.store.get(id, {}).get(key, default)

    def reset_config(self, id: TrialId, key: str = None) -> None:
        if id in self.store:
            if key is None:
                del self.store[id]
            elif key in self.store[id]:
                del self.store[id][key]
