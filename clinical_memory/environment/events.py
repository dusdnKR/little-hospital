def event_exams(event):
    if "exams" in event:
        return event["exams"]
    return [event["exam"]] if "exam" in event else []
