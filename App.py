import streamlit as st

from GPTAssistant import GPTAssistant
from datetime import datetime
import pytz

st.write("## 회의실 자동예약 도우미")
st.write("Streamlit 라이브러리의 특성상 화면이 불안정할 수 있습니다.")
st.markdown(
    """
    <div style="margin-bottom: 50px;"></div>
""",
    unsafe_allow_html=True,
)


OPENAI_KEY = st.secrets["OPENAI_KEY"]
kst = pytz.timezone("Asia/Seoul")

today = datetime.now().astimezone(kst)
str_tdy = today.strftime("%A, %B %d, %Y")
str_time = today.strftime("%H:%M")


if "meeting_rooms" not in st.session_state:
    st.session_state.meeting_rooms = [
        "회의실1",
        "회의실2",
        "문화복합콤플렉스",
    ]

if "new_item" not in st.session_state:
    st.session_state.new_item = ""

if "conv_template" not in st.session_state:
    st.session_state.conv_template = [
        {
            "speaker": "김지영 대리",
            "text": "대표님이 {when}에 우리 팀 다같이 보자시네요",
        },
        {"speaker": "권은서 주임", "text": "헐 무섭다 회의실 예약할까요?"},
        {"speaker": "You", "text": "왜 부르시는지는 모르죠?"},
        {
            "speaker": "김지영 대리",
            "text": "네 이유는 모르겠네요;; 일단 {where} 잡아주세요",
        },
    ]

if "variables" not in st.session_state:
    st.session_state.variables = {"when": "30분 뒤", "where": "복합 어쩌고? 거기"}

if "conv_updated" not in st.session_state:
    st.session_state.conv_updated = [
        {
            "speaker": entry["speaker"],
            "text": entry["text"].format(**st.session_state.variables),
        }
        for entry in st.session_state.conv_template
    ]

if "result" not in st.session_state:
    st.session_state.result = None


def add_item(item):
    new_item = item.strip()
    if new_item and new_item not in st.session_state.meeting_rooms:
        st.session_state.meeting_rooms.append(new_item)


def remove_item(item):
    if len(st.session_state.meeting_rooms) > 2:  # 최소 길이 유지
        st.session_state.meeting_rooms.remove(item)
        st.rerun()


def chat_message(speaker, text):
    bg_color = "#FFFB81" if speaker == "You" else "#F0F2F6"
    align = "right" if speaker == "You" else "left"

    # HTML + CSS로 말풍선 스타일 적용
    chat_html = f"""
    <div style='text-align: {align}; margin: 10px 0;'>
        <div style='font-size: 12px; color: #555; margin-bottom: 2px;'>{speaker}</div>
        <div style='display: inline-block; font-size: 14px; padding: 10px 15px; border-radius: 10px; 
                    background-color: {bg_color}; max-width: 80%; text-align: left;'>
            {text}
        </div>
    </div>
    """
    st.markdown(chat_html, unsafe_allow_html=True)


col1, col2 = st.columns(2, gap="medium")

with col1:
    st.write("#### 1. 회의실 목록 설정")
    new_item = st.text_input("새 회의실 추가(엔터로 추가)")
    if new_item:
        add_item(new_item)
        
    for item in st.session_state.meeting_rooms:
        cols = st.columns([4, 1])
        cols[0].write(item)
        # if len(st.session_state.meeting_rooms) > 2:  # 최소 길이 제한
        #     if cols[1].button("삭제", key=f"remove_{item}"):
        #         remove_item(item)

    


tools = today = datetime.now().astimezone(kst)
str_tdy = today.strftime("%A, %B %d, %Y")
str_time = today.strftime("%H:%M")


tools = [
    {
        "type": "function",
        "function": {
            "name": "reserve_meetingroom",
            "description": "Based on the list of entered conversations, identify the meeting location and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "meeting_room": {
                        "type": "string",
                        "description": "Meeting location inferred from the conversation list.",
                        "enum": st.session_state.meeting_rooms,
                    },
                    "meeting_date": {
                        "type": "string",
                        "description": f"Meeting date inferred from the conversation list and display it in YYYY-mm-dd format. Today’s date is {str_tdy}. If a relative date reference (e.g., '30분 뒤', '내일', '다음 주') is mentioned, calculate the exact date accordingly. If a specific day is inferred but is earlier than today’s date, assume it refers to the same day in the next month. Ensure the date is valid by considering the last day of each month (e.g., January 31, February 28/29, March 31, etc.).",
                    },
                    "meeting_time": {
                        "type": "string",
                        "description": f"Meeting time inferred from the conversation list, displayed in 24-hour format (e.g., 17:30). The current time is {str_time}. If the time reference is ambiguous (e.g., no AM/PM indication), infer it as AM from 09:00 to 11:59 and as PM otherwise. If a relative time (e.g., '30분 뒤', '1시간 뒤') is mentioned, calculate the exact time accordingly. If no specific time is mentioned and no relative time reference exists, return 09:00.",
                    },
                    "meeting_topic": {
                        "type": "string",
                        "description": "Meeting topic inferred from the conversation list: within 30 characters, in 한국어. e.g., 'OO 사업 기획 관련 회의', 'OO사 미팅', etc.",
                    },
                },
                "required": [
                    "meeting_room",
                    "meeting_date",
                    "meeting_time",
                    "meeting_topic",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

concept = "You are a auto meetingroom reservation bot. Based on the list of entered conversations, identify the meeting location and time."

assistant = GPTAssistant(
    api_key=OPENAI_KEY,
    model_name="gpt-4o-mini",
    instructions=concept,
    tools=tools,
    return_args_only=True,
)

with col2:
    st.write("#### 2. 채팅 내용 수정")
    st.session_state.variables["when"] = st.text_input(
        "회의 날짜/시간:", value=st.session_state.variables["when"]
    )
    st.session_state.variables["where"] = st.text_input(
        "회의실 위치:", value=st.session_state.variables["where"]
    )
    if st.button("업데이트"):
        st.session_state.conv_updated = [
            {
                "speaker": entry["speaker"],
                "text": entry["text"].format(**st.session_state.variables),
            }
            for entry in st.session_state.conv_template
        ]
        st.rerun()


st.divider()
result_1, result_2 = st.columns(2)

with result_1:
    st.write("#### 3. 채팅 확인")
    for msg in st.session_state.conv_updated:
        chat_message(msg["speaker"], msg["text"])

with result_2:
    st.write("#### 4. 실행")
    if st.button("실행"):
        with st.spinner("처리 중..."):
            result = assistant.first_chat(str(st.session_state.conv_updated))
            st.session_state.result = result["results"][0]["output"]

    if st.session_state.result:
        st.code(st.session_state.result, language="json", wrap_lines=True)
