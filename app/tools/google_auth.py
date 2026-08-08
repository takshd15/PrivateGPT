from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.config import GOOGLE_CREDENTIALS_FILE, GOOGLE_TOKEN_FILE

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
]


def _has_required_scopes(creds: Credentials) -> bool:
    granted = set(creds.scopes or [])
    return set(SCOPES).issubset(granted)


def _delete_saved_token() -> None:
    try:
        GOOGLE_TOKEN_FILE.unlink()
    except FileNotFoundError:
        pass


def _run_consent_flow() -> Credentials:
    flow = InstalledAppFlow.from_client_secrets_file(
        str(GOOGLE_CREDENTIALS_FILE),
        SCOPES,
    )
    return flow.run_local_server(port=0)


def _save_credentials(creds: Credentials) -> None:
    GOOGLE_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    GOOGLE_TOKEN_FILE.write_text(creds.to_json())


def get_google_credentials():
    creds = None

    if GOOGLE_TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(GOOGLE_TOKEN_FILE))

    if creds and not _has_required_scopes(creds):
        _delete_saved_token()
        creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                _delete_saved_token()
                creds = _run_consent_flow()
        else:
            creds = _run_consent_flow()

        _save_credentials(creds)

    return creds


def gmail_service():
    return build("gmail", "v1", credentials=get_google_credentials())


def calendar_service():
    return build("calendar", "v3", credentials=get_google_credentials())
