"""Interface translations only. Recorded clinical dialogue remains untouched."""

STRINGS = {
    "subtitle": ("의료 진단 시뮬레이션", "Medical diagnosis simulation"),
    "day_shift": ("낮 근무", "Day shift"),
    "night_shift": ("밤 근무", "Night shift"),
    "coins": ("코인", "COINS"),
    "deaths": ("사망", "DEATHS"),
    "survival": ("생존률", "SURVIVAL"),
    "counted_rooms": ("01–03 진료실만 포함", "Rooms 01–03 only"),
    "pause": ("일시정지", "Pause"),
    "paused": ("일시정지", "Paused"),
    "resume_hint": (
        "클릭하거나 SPACE를 누르면 계속합니다",
        "Click or press SPACE to resume",
    ),
    "count": ("{done} / {total} 진료", "{done} / {total} visits"),
    "home": ("전체 병원", "Hospital"),
    "sources": ("데이터 출처", "Data sources"),
    "chat": ("대화 보기", "Conversations"),
    "history": ("진료 기록", "Visit records"),
    "overview": ("Little Hostpial", "Little Hostpial"),
    "waiting": ("다음 환자 대기", "Awaiting a patient"),
    "card_count": ("회복 {wins} / 진료 {visits}", "Recovered {wins} / Visits {visits}"),
    "controls": ("조작 방법", "Controls"),
    "program": ("프로그램 안내", "About the program"),
    "key_pause": ("일시정지 / 재개", "Pause / Resume"),
    "key_rooms": ("진료실 선택", "Select a clinic"),
    "key_back": ("돌아가기", "Go back"),
    "key_scroll": ("대화 / 기록 스크롤", "Scroll conversations"),
    "wheel": ("마우스 휠", "MOUSE WHEEL"),
    "image_sources": (
        "검사 이미지 · MedXpertQA 데이터셋",
        "Examination images · MedXpertQA dataset",
    ),
    "welcome": ("접수가 시작됩니다", "Reception is open"),
    "transcript": ("진료 대화", "Consultation"),
    "previous": ("이전 진료", "Past visits"),
    "waiting_patient": ("환자를 기다리고 있습니다.", "Waiting for a patient."),
    "arrival_route": (
        "접수 → 대기 → 진료실 입장",
        "Reception → Waiting → Consultation",
    ),
    "finished": ("진료 완료", "Completed"),
    "record_phase": ("실험 구간: {phase}", "Phase: {phase}"),
    "speaking": ("{name} 발언 중 {dots}", "{name} is speaking {dots}"),
    "scroll_chat": ("휠로 이전 대화 확인", "Scroll to read earlier messages"),
    "latest": ("최신 대화", "Latest"),
    "history_hint": (
        "완료된 진료의 결과와 대화 · 병원 안의 시뮬레이션 시각",
        "Completed visits and conversations · Hospital simulation time",
    ),
    "json": ("JSON 내보내기", "Export JSON"),
    "csv": ("CSV 요약 내보내기", "Export CSV summary"),
    "close": ("닫기", "Close"),
    "all": ("전체", "All"),
    "recovered": ("회복", "Recovered"),
    "dead": ("사망", "Died"),
    "record_count": (
        "{count}개 진료 · 휠로 목록 이동",
        "{count} visits · Scroll the list",
    ),
    "paid": ("결제 완료", "Paid"),
    "pending_payment": ("결제 대기", "Awaiting payment"),
    "source_file": ("원본: {path}", "Source: {path}"),
    "source_time": (
        "실험 구간: {phase} · 원본 일시: {time}",
        "Phase: {phase} · Recorded at: {time}",
    ),
    "unrecorded": ("미기록", "Not recorded"),
    "empty_history": ("아직 완료된 진료가 없습니다.", "No completed visits yet."),
    "history_later": (
        "진료 결과가 확정되면 이곳에 저장됩니다.",
        "Completed visits will appear here.",
    ),
    "replay_data": ("재생 데이터", "Replay data"),
    "help_title": ("Little Hospital 사용 안내", "Welcome to Little Hospital"),
    "source_intro": (
        "저장된 대화와 판정을 재생합니다. 모델이나 네트워크를 호출하지 않습니다.",
        "Plays recorded conversations and outcomes without model or network calls.",
    ),
    "model_records": ("{model} · {count}개 기록", "{model} · {count} records"),
    "source_config": (
        "sources.json 또는 --config로 기록 경로를 변경할 수 있습니다.",
        "Change record paths in sources.json or with --config.",
    ),
    "saved": ("{count}개 기록 저장 완료", "Saved {count} visits"),
    "export_error": ("내보내기 실패", "Export failed"),
    "file_type": ("진료 기록", "Visit records"),
    "exam_room": ("검사실", "EXAMINATION"),
    "exam_hint": ("한 팀씩 이용", "One team at a time"),
    "examining": ("검사 중", "Examination"),
    "reception_1": ("접수 01", "RECEPTION 01"),
    "reception_2": ("접수 02", "RECEPTION 02"),
    "payment": ("수납", "CHECKOUT"),
    "image_view": ("검사 이미지", "Examination image"),
    "image_close": ("닫기 / ESC", "Close / ESC"),
    "assigned": ("{name} → {doctor}", "{name} → {doctor}"),
    "exam_received": ("{doctor} · 검사 결과 도착", "{doctor} · Examination received"),
    "outcome": ("{doctor} · {name} · {result}", "{doctor} · {name} · {result}"),
    "payment_event": ("{name} 결제 · +1 코인", "{name} paid · +1 coin"),
    "complete": ("모든 진료가 끝났습니다", "All visits are complete"),
}
STATES = dict(
    zip(
        (
            "입장",
            "접수 대기",
            "접수",
            "벤치 대기",
            "배정 대기",
            "진료실 이동",
            "문진 중",
            "진단 중",
            "검사실 대기",
            "검사 이동",
            "검사 중",
            "진료실 복귀",
            "회복",
            "영혼",
            "결제 대기",
            "결제",
            "퇴장",
        ),
        (
            "Arriving",
            "Reception queue",
            "Checking in",
            "Waiting lounge",
            "Awaiting a room",
            "Entering clinic",
            "Consulting",
            "Diagnosing",
            "Awaiting examination",
            "Going to examination",
            "Examination",
            "Returning to clinic",
            "Recovered",
            "Departed",
            "Payment queue",
            "Checking out",
            "Leaving",
        ),
    )
)
HELP = {
    "ko": {
        "program": [
            "세 Qwen 의사의 진료 과정을 픽셀 병원에서 재생합니다. 환자는 접수·대기·진료·검사를 거쳐 회복하면 수납 후 퇴장합니다. 배정되는 환자의 상체와 방 번호가 접수 말풍선과 복도 왼쪽 벽걸이 TV에 함께 나타납니다.",
            "코인은 수납할 때 1개씩 증가합니다. 생존률은 01–03 진료실의 완료 진료 중 회복 비율입니다. 저장된 정답 여부를 회복·사망으로 표현한 게임 지표입니다.",
        ],
        "controls": [
            "SPACE 일시정지·재개 / 숫자 1–3 진료실 선택 / ESC 돌아가기 / 마우스 휠 대화·기록·도움말 스크롤. 진료실이나 ‘대화 보기’를 누르면 대화창이 열립니다.",
            "기본 속도는 1×이며 0.5×·2×·4×로 바꿀 수 있습니다. KO / EN으로 언어를 전환합니다. 받은 검사 이미지는 채팅 상단에도 남으며 클릭하면 확대됩니다.",
            "‘진료 기록’에서 완료된 대화를 확인하고 JSON/CSV로 저장합니다. RESET은 카운터·진료 목록을 비우고 1×로 시작합니다. 필요한 기록은 RESET이나 종료 전에 내보내세요.",
        ],
    },
    "en": {
        "program": [
            "Follow three Qwen doctors in a pixel hospital. Patients check in, wait, consult, and visit the examination room. Recovered patients pay before leaving. The reception bubble and the TV on the left corridor wall show the patient portrait and assigned room number.",
            "Each checkout adds one coin. Survival is the recovery share of completed visits in rooms 01–03. Game recovery and death represent the correctness of recorded diagnoses.",
        ],
        "controls": [
            "SPACE pauses or resumes; 1–3 select clinics; ESC goes back; the mouse wheel scrolls conversations, records, and help. Click a clinic or Conversations to open a chat.",
            "Playback starts at 1×. Choose 0.5×, 2×, or 4×, and switch language with KO / EN. Received examination images remain above the conversation. Click a thumbnail to enlarge it.",
            "Visit records contains completed conversations and JSON/CSV exports. RESET clears counters and records, then starts at 1×. Export needed records before resetting or closing.",
        ],
    },
}


def tr(language, key, **values):
    return STRINGS[key][language == "en"].format(**values)


def state_text(language, state):
    return STATES.get(state, state) if language == "en" else state
