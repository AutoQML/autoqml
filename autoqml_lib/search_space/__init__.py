from typing import Union

from ray.tune.search.sample import Domain

SearchSpace = dict[str, Domain]
Configuration = dict[str, Union[str, float, int, bool, None]]


def prune_search_space(
    search_space: SearchSpace, user_values: Configuration
) -> SearchSpace:
    # 1. Replace each search range in SearchSpace with user provided value
    # 2. Can be either constant value or adapted Domain
    # 3. Return pruned search_space
    pass
