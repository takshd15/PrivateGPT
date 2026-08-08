' Jarvix v2 - launch the wake loop fully hidden (no console window) on login.
' Uses pythonw.exe from the venv so nothing flashes on screen.
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "C:\Users\hp\VectorDB\jarvix"
sh.Run """C:\Users\hp\VectorDB\jarvix\.venv\Scripts\pythonw.exe"" -m app.runtime.startup", 0, False
