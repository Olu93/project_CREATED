


# %%
from typing import Any, Generic, TypeVar,Type

T = TypeVar("T")
S = TypeVar("S")

class Foo:
    pass

class FooBar(Foo):
    bar = None
    def __init__(self) -> None:
        # super().__init__()
        print("FooBar")
    
class FooFoo(Foo):
    foo = None
    def __init__(self) -> None:
        # super().__init__()
        print("FooFoo")

    
class FinalClass(Generic[S]):
    def __init__(self) -> None:
        SomeType = ToInstantiate(S)
        print(SomeType)
        super().__init__()

class ToInstantiate():
    def __new__(self, T:Type[FinalClass]):
        if T==FooFoo:
            return FooFoo()
        if T==FooBar:
            return FooBar()
        return Foo()

obj = FinalClass[FooBar]()
print(obj)
# %%
