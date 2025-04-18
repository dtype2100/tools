import streamlit as st
import requests, time

API_BASE = "http://localhost:8000"

st.title("🤖 Hugging Face / GGUF 모델 관리자")

model_name = st.text_input("모델 이름 또는 경로")
save_path = st.text_input("저장 경로 (옵션)", value="") or None
model_format = st.selectbox("모델 포맷", options=["transformers", "gguf"])

def post_to_api(endpoint):
    return requests.post(f"{API_BASE}/{endpoint}", json={
        "model_name": model_name,
        "save_path": save_path,
        "model_format": model_format
    })

def handle_response(res):
    try:
        data = res.json()
        if "message" in data:
            st.success(data["message"])
        elif "error" in data:
            st.error(data["error"])
        elif "detail" in data:
            st.error(data["detail"])
        else:
            st.warning("예상치 못한 응답입니다.")
    except Exception as e:
        st.error(f"응답 파싱 오류: {e}")

if st.button("📥 모델 다운로드"):
    with st.spinner("모델 다운로드 중..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        res = post_to_api("download")
    handle_response(res)

if st.button("📦 모델 로드"):
    res = post_to_api("load")
    handle_response(res)

if st.button("🗑 모델 삭제"):
    res = post_to_api("delete")
    handle_response(res)

prompt = st.text_area("📝 프롬프트 입력")
if st.button("🚀 추론 요청"):
    res = requests.post(f"{API_BASE}/infer", params={"prompt": prompt}, json={
        "model_name": model_name
    })
    handle_response(res)

st.markdown("---")
st.subheader("📚 현재 다운로드된 모델 목록")
if st.button("🔄 모델 목록 새로고침"):
    res = requests.get(f"{API_BASE}/list_models")
    models = res.json().get("models", [])
    if models:
        st.write(models)
    else:
        st.info("저장된 모델이 없습니다.")
    