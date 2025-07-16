from typing import Type, TypeVar


T = TypeVar("T")


def _only_overlap_as_params(x: dict, Params: Type[T]) -> T:
    return Params(
        **{
            key: value
            for key, value in x.items()
            if key in Params.__annotations__  # type: ignore
        }
    )


def only_overlap(x: dict, Params: Type[T]) -> T:
    return {
        key: value
        for key, value in x.items()
        if key in Params.__required_keys__  # type: ignore
    }
