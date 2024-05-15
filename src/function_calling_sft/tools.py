from function_calling_sft.schema import ArgumentSpec, ToolRegistry, ToolSpec

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")


def default_registry() -> ToolRegistry:
    return ToolRegistry(
        (
            ToolSpec(
                "get_weather",
                "Read deterministic local weather",
                {
                    "city": ArgumentSpec("string", "City name"),
                    "unit": ArgumentSpec(
                        "string", "Temperature unit", enum=("celsius", "fahrenheit")
                    ),
                },
            ),
            ToolSpec(
                "schedule_meeting",
                "Add a meeting to the local calendar",
                {
                    "title": ArgumentSpec("string", "Meeting title"),
                    "day": ArgumentSpec("string", "Weekday", enum=DAYS),
                    "start_hour": ArgumentSpec("integer", "Start hour", minimum=0, maximum=23),
                    "duration_minutes": ArgumentSpec(
                        "integer", "Duration", minimum=15, maximum=240
                    ),
                },
            ),
            ToolSpec(
                "set_reminder",
                "Create a local reminder",
                {
                    "task": ArgumentSpec("string", "Task to remember"),
                    "day": ArgumentSpec("string", "Weekday", enum=DAYS),
                    "hour": ArgumentSpec("integer", "Reminder hour", minimum=0, maximum=23),
                },
            ),
        )
    )
