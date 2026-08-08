DANGEROUS_ACTIONS = {
    "send_email",
    "delete_email",
    "archive_email",
    "create_calendar_event",
    "delete_calendar_event",
    "move_file",
    "delete_file",
    "run_terminal_command",
    # Structured-capture writes (app/brain/structuring.py + dialogue.py). Listed
    # here as an audit marker only - confirmation for these is always the
    # spoken/typed "yes" in dialogue.py's *_confirm states, never a second
    # blocking permissions.confirm() prompt (same as create_calendar_event).
    "remember_task",
    "new_project",
    "log_progress_event",
}


def needs_confirmation(action_name: str) -> bool:
    return action_name in DANGEROUS_ACTIONS


def confirm(action_description: str) -> bool:
    print("\nJarvix wants to do this:")
    print(action_description)
    answer = input("Allow? Type yes/no: ").strip().lower()
    return answer == "yes"
