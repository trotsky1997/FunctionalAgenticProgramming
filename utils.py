from typing import Any, Callable
import inspect
from typing import Callable, Any
import warnings

import astroid
import astypes
from pydantic import create_model


def get_schema_from_signature(fn: Callable) -> str:
    """Turn a function signature into a JSON schema.

    Every JSON object valid to the output JSON Schema can be passed
    to `fn` using the ** unpacking syntax.

    """
    code = inspect.getsource(fn)

    signature = inspect.signature(fn)
    arguments = {}
    for name, arg in signature.parameters.items():
        if arg.annotation == inspect._empty:
            if arg.default != inspect._empty:
                node = astroid.extract_node(str(arg.default))
                inferred_type = astypes.get_type(node)
                arguments[name] = (inferred_type, ...)
                continue
            else:
                arguments[name] = (Any, ...)
                continue
        else:
            arguments[name] = (arg.annotation, ...)

    try:
        fn_name = fn.__name__
    except Exception as e:
        fn_name = "Arguments"
        warnings.warn(
            f"The function name could not be determined. Using default name 'Arguments' instead. For debugging, here is exact error:\n{e}",
            category=UserWarning,
        )
    model = create_model(fn_name, **arguments)

    return model.model_json_schema()


def get_return_schema_from_signature(fn: Callable) -> str:
    """Turn a function signature into a JSON schema.

    Every JSON object valid to the output JSON Schema can be passed
    to `fn` using the ** unpacking syntax.

    """
    code = inspect.getsource(fn)

    signature = inspect.signature(fn)

    # Get the return annotation
    return_annotation = signature.return_annotation
    if return_annotation == inspect._empty:
        # If there's no return annotation, set it to Any
        return_annotation = Any
    return_dict = {f'return of {fn.__name__}': (return_annotation, ...)}
    return create_model("return_dict", **return_dict).model_json_schema()

