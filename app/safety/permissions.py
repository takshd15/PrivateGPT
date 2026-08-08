DANGEROUS_ACTIONS = {
    "send_email",
    "delete_email",
    "archive_email",
    "create_calendar_event",
    "delete_calendar_event",
    "move_file",
    "delete_file",
    "run_terminal_command",
}


def needs_confirmation(action_name: str) -> bool:
    return action_name in DANGEROUS_ACTIONS


def confirm(action_description: str) -> bool:
    print("\nJarvix wants to do this:")
    print(action_description)
    answer = input("Allow? Type yes/no: ").strip().lower()
    return answer == "yes"
