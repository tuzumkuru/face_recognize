from datetime import datetime


def notify_match(name: str, distance: float):
    ts = datetime.now().isoformat()
    print(f"[MATCH] {ts} - {name} (distance={distance:.3f})")


def notify_unknown(distance: float):
    ts = datetime.now().isoformat()
    print(f"[UNKNOWN] {ts} - unknown face (closest distance={distance:.3f})")
