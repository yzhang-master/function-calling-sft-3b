from function_calling_sft import ArgumentSpec, ToolSpec, validate_arguments

tool = ToolSpec("ping", "Check a host", {"host": ArgumentSpec("string", "Host name")})
print(validate_arguments(tool, {"host": "localhost"}))
