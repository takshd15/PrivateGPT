import base64
from app.tools.google_auth import gmail_service


def _decode_body(payload: dict) -> str:
    body_data = ""

    if "body" in payload and payload["body"].get("data"):
        body_data = payload["body"]["data"]

    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            body_data = part["body"]["data"]
            break

    if not body_data:
        return ""

    decoded = base64.urlsafe_b64decode(body_data.encode("utf-8"))
    return decoded.decode("utf-8", errors="ignore")


def get_recent_emails(limit: int = 10, days: int = 7) -> list[dict]:
    service = gmail_service()

    results = (
        service.users()
        .messages()
        .list(
            userId="me",
            maxResults=limit,
            q=f"newer_than:{days}d",
        )
        .execute()
    )

    messages = results.get("messages", [])
    emails = []

    for msg in messages:
        full = (
            service.users()
            .messages()
            .get(userId="me", id=msg["id"], format="full")
            .execute()
        )

        headers = full.get("payload", {}).get("headers", [])
        header_map = {h["name"].lower(): h["value"] for h in headers}

        emails.append(
            {
                "id": full["id"],
                "from": header_map.get("from", ""),
                "subject": header_map.get("subject", ""),
                "date": header_map.get("date", ""),
                "snippet": full.get("snippet", ""),
                "body": _decode_body(full.get("payload", {}))[:3000],
            }
        )

    return emails
