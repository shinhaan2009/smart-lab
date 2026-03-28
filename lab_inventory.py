import streamlit as st
import pandas as pd
import time
import re
import numpy as np
from PIL import Image
import easyocr
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -----------------------------------------------------------------------------
# [1. 초기 설정 및 세션 데이터]
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SmartLab AI + Map", page_icon="🧪", layout="wide")

# CAS 기반 위험성 DB
CAS_DATABASE = {
    "11118-57-3": {
        "name": "Chromium oxide",
        "hazard_level": "DANGER",
        "msg": "⚠️ 고위험 물질 — 보호구 착용 및 후드 내 취급."
    },
    "5350-57-2": {
        "name": "Benzophenone hydrazone",
        "hazard_level": "UNKNOWN",
        "msg": "정보 부족 — MSDS 확인 필요."
    },
    "7664-93-9": {"name": "황산 (Sulfuric Acid)", "hazard_level": "DANGER", "msg": "⚠️ 피부 부식성/화상 위험! 보호구 착용 필수."},
    "67-56-1": {"name": "메탄올 (Methanol)", "hazard_level": "WARNING", "msg": "🔥 인화성 및 독성. 환기 필요."},
    "64-17-5": {"name": "에탄올 (Ethanol)", "hazard_level": "WARNING", "msg": "🔥 인화성 물질. 화기 엄금."},
    "7647-01-0": {"name": "염산 (Hydrochloric Acid)", "hazard_level": "DANGER", "msg": "☠️ 유독 가스 발생 가능. 후드 내 사용."},
    "7732-18-5": {"name": "증류수 (Distilled Water)", "hazard_level": "SAFE", "msg": "✅ 안전한 물질입니다."}
}

CATEGORIES = ["초자류", "소모품", "기기 및 장비", "시약 및 화학물질", "안전 용품"]

# 상호작용 경고
INTERACTION_WARNINGS = [
    (("황산", "염산"), "강산-강산 조합: 산성 환경에서 가열 시 위험 증가."),
    (("황산", "에탄올"), "황산 + 에탄올: 발열성 에스터화 혹은 분해 반응 가능 — 엄격한 주의 필요."),
    (("과산화수소", "유기용매"), "과산화물 형성 및 폭발 위험 가능성 — 격리 보관 권장."),
    (("시안", "산"), "시안 화합물 + 산: 유독가스(HCN) 발생 가능 — 치명적 위험."),
]

# 실험 레시피
RECIPES = {
    "DNA 추출 (에탄올 침전)": {
        "description": "간단한 DNA 추출 후 에탄올을 이용한 침전 방식",
        "reagents": [
            {"name": "에탄올 (Ethanol)", "cas": "64-17-5", "amount": "70% 500 mL"},
            {"name": "TE buffer", "cas": "-", "amount": "50 mL"},
            {"name": "EDTA", "cas": "6381-92-6", "amount": "0.5 M 1 mL"},
            {"name": "NaCl", "cas": "7647-14-5", "amount": "5 M 1 mL"}
        ],
        "substitutes": {"에탄올 (Ethanol)": "IPA (Isopropanol - 67-63-0)"}
    },
    "SDS-PAGE 샘플 준비": {
        "description": "단백질 샘플을 변성시키기 위한 버퍼 및 환원제 포함",
        "reagents": [
            {"name": "Tris-HCl", "cas": "1185-53-1", "amount": "1 M 1 mL"},
            {"name": "SDS (Sodium dodecyl sulfate)", "cas": "151-21-3", "amount": "10% 1 mL"},
            {"name": "β-mercaptoethanol", "cas": "60-24-2", "amount": "5% 100 μL"}
        ],
        "substitutes": {"β-mercaptoethanol": "DTT (Dithiothreitol - 3483-12-3)"}
    },
    "LB 배지 준비": {
        "description": "세균 배양용 LB 배지를 만들기 위한 기본 시약",
        "reagents": [
            {"name": "Tryptone", "cas": "9002-07-7", "amount": "10 g"},
            {"name": "Yeast extract", "cas": "8013-01-2", "amount": "5 g"},
            {"name": "NaCl", "cas": "7647-14-5", "amount": "10 g"}
        ],
        "substitutes": {}
    },
    "산-염기 중화(적정 실습)": {
        "description": "산-염기 반응을 보여주는 간단한 실험",
        "reagents": [
            {"name": "염산 (Hydrochloric Acid)", "cas": "7647-01-0", "amount": "0.1 M 50 mL"},
            {"name": "수산화나트륨 (Sodium hydroxide)", "cas": "1310-73-2", "amount": "0.1 M 50 mL"},
            {"name": "페놀프탈레인 지시약", "cas": "-", "amount": "소량"}
        ],
        "substitutes": {"염산 (Hydrochloric Acid)": "구연산(Citric acid - 77-92-9)로 저위험 시연 가능"}
    }
}

# 세션 상태 초기화
if 'inventory' not in st.session_state:
    st.session_state.inventory = [
        {"id": 1, "name": "황산 (98%)", "category": "시약 및 화학물질", "quantity": 2, "unit": "L",
         "cas": "7664-93-9", "hazard": "DANGER", "status": "양호", "location": "Area A"},
    ]
if 'lab_map' not in st.session_state:
    st.session_state.lab_map = [
        {"구역명": "Area A", "x": 50, "y": 50, "w": 150, "h": 200}
    ]
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

if "last_deleted" not in st.session_state:
    st.session_state.last_deleted = []


# -----------------------------------------------------------------------------
# [2. 유틸리티 함수]
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['en'], gpu=False)


def extract_cas_with_easyocr(image):
    try:
        reader = load_ocr_model()
        img_np = np.array(image)
        result_list = reader.readtext(img_np, detail=0)
        full_text = " ".join(result_list)
        cas_pattern = re.compile(r'\b\d{2,7}-\d{2}-\d\b')
        match = cas_pattern.search(full_text)
        return full_text, match.group() if match else None
    except Exception as e:
        return f"Error: {str(e)}", None


def find_cas_by_name(name):
    name_l = name.lower()
    for cas, info in CAS_DATABASE.items():
        if info.get('name') and name_l in info['name'].lower():
            return cas
    return None


def get_hazard_info_by_cas_or_name(cas, name):
    if cas and cas in CAS_DATABASE:
        return CAS_DATABASE[cas]
    if name:
        found = find_cas_by_name(name)
        if found:
            return CAS_DATABASE[found]
    return {"name": name or "미등록 물질", "hazard_level": "UNKNOWN", "msg": "정보 없음"}


def check_interactions(reagents):
    warnings = []
    names = [r['name'] for r in reagents]
    for pair, msg in INTERACTION_WARNINGS:
        a_kw, b_kw = pair
        if any(a_kw in s for s in names) and any(b_kw in s for s in names):
            warnings.append(msg)
    return list(set(warnings))


def inventory_has_item(reagent):
    cas = reagent.get('cas')
    name = reagent.get('name', '').lower()
    for it in st.session_state.inventory:
        if cas != '-' and cas and it.get('cas') == cas:
            return True
        if name and name in it.get('name', '').lower():
            return True
    return False


# -----------------------------------------------------------------------------
# [3. 사이드바 메뉴]
# -----------------------------------------------------------------------------
st.sidebar.title("🧪 SmartLab Pro")
menu = st.sidebar.radio("메뉴 이동", ["📊 대시보드", "📦 재고 관리", "📷 AI 안전 스캐너", "🔬 실험 레시피", "🗺️ 실험실 지도 설정"])

# 현재 등록된 구역 리스트
available_locations = [area['구역명'] for area in st.session_state.lab_map] if st.session_state.lab_map else ["미지정"]

# -----------------------------------------------------------------------------
# [4. 각 메뉴별 구현]
# -----------------------------------------------------------------------------

# --- 1. 대시보드 ---
if menu == "📊 대시보드":
    st.title("실험실 안전 대시보드")

    inv_df = pd.DataFrame(st.session_state.inventory)
    col1, col2, col3 = st.columns(3)
    col1.metric("전체 물품", f"{len(inv_df)}건")
    col2.metric("위험 물질", f"{len(inv_df[inv_df['hazard'] == 'DANGER'])}건")
    col3.metric("최근 업데이트", datetime.now().strftime("%Y-%m-%d"))

    st.divider()

    st.subheader("🖼️ 실시간 물품 배치 지도")
    if st.session_state.lab_map:
        CANVAS_WIDTH, CANVAS_HEIGHT = 800, 450
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_xlim(0, CANVAS_WIDTH)
        ax.set_ylim(CANVAS_HEIGHT, 0)
        ax.set_facecolor('#f0f2f6')

        for area in st.session_state.lab_map:
            # 해당 구역에 있는 물품 검색
            stored_items = inv_df[inv_df['location'] == area['구역명']]['name'].tolist() if not inv_df.empty else []
            items_text = "\n".join(stored_items[:3]) + ("..." if len(stored_items) > 3 else "")

            rect = patches.Rectangle((area['x'], area['y']), area['w'], area['h'],
                                     linewidth=2, edgecolor='#31333F', facecolor='#A0C4FF', alpha=0.6)
            ax.add_patch(rect)

            display_label = f"[{area['구역명']}]\n{items_text if items_text else '(비어 있음)'}"
            ax.text(area['x'] + area['w'] / 2, area['y'] + area['h'] / 2, display_label,
                    ha='center', va='center', fontsize=10, fontweight='bold', color='black')

        plt.title("Lab Real-time Inventory Map")
        st.pyplot(fig)
    else:
        st.info("🗺️ '실험실 지도 설정' 메뉴에서 먼저 구역을 그려주세요.")

# --- 2. 재고 관리 ---
elif menu == "📦 재고 관리":
    st.title("전체 재고 관리")

    df = pd.DataFrame(st.session_state.inventory)
    if df.empty:
        st.info("현재 재고가 비어 있습니다.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("➕ 수동 품목 추가"):
        with st.form("add_form"):
            c1, c2 = st.columns(2)
            name = c1.text_input("물품명")
            cat = c2.selectbox("카테고리", CATEGORIES)
            loc = st.selectbox("보관 위치 (지도에서 설정한 구역)", available_locations)

            if st.form_submit_button("추가하기"):
                new_item = {
                    "id": int(time.time() * 1000),
                    "name": name,
                    "category": cat,
                    "quantity": 1,
                    "unit": "EA",
                    "cas": "-",
                    "hazard": "SAFE",
                    "status": "양호",
                    "location": loc,
                }
                st.session_state.inventory.append(new_item)
                st.success(f"'{name}' 품목이 추가되었습니다.")
                st.rerun()

    with st.expander("➖ 수동 품목 제거"):
        inv = st.session_state.inventory
        if not inv:
            st.info("삭제할 품목이 없습니다.")
        else:
            display_list = [
                f"{it['name']} | {it['category']} | {it['quantity']}{it.get('unit', '')} | {it.get('location', '-')} — id:{it['id']}"
                for it in inv]

            cols = st.columns([3, 1])
            with cols[0]:
                to_delete = st.multiselect("삭제할 품목 선택", display_list)
            with cols[1]:
                if st.button("선택 품목 삭제"):
                    if not to_delete:
                        st.warning("선택된 항목이 없습니다.")
                    else:
                        ids_to_remove = [int(s.split("id:")[-1]) for s in to_delete]
                        removed = [it for it in st.session_state.inventory if it['id'] in ids_to_remove]
                        st.session_state.inventory = [it for it in st.session_state.inventory if
                                                      it['id'] not in ids_to_remove]
                        st.session_state.last_deleted = removed
                        st.success(f"{len(removed)}개 품목을 삭제했습니다.")
                        st.rerun()

            st.markdown("---")
            with st.form("delete_by_id_form"):
                id_col1, id_col2 = st.columns([2, 1])
                raw_id = id_col1.text_input("ID 직접 입력")
                if id_col2.form_submit_button("ID로 삭제"):
                    try:
                        raw_id_int = int(raw_id)
                        matching = [it for it in st.session_state.inventory if it['id'] == raw_id_int]
                        if not matching:
                            st.warning("해당 ID 없음")
                        else:
                            st.session_state.inventory = [it for it in st.session_state.inventory if
                                                          it['id'] != raw_id_int]
                            st.session_state.last_deleted = matching
                            st.success("삭제 완료")
                            st.rerun()
                    except:
                        st.error("올바른 숫자 ID를 입력하세요.")

    if st.session_state.last_deleted:
        with st.expander("↩️ 최근 삭제 취소"):
            for it in st.session_state.last_deleted:
                st.write(f"• {it['name']}")
            if st.button("삭제 취소 (되돌리기)"):
                st.session_state.inventory.extend(st.session_state.last_deleted)
                st.session_state.last_deleted = []
                st.success("복원되었습니다.")
                st.rerun()

    with st.expander("🔽 추가 작업"):
        c_export, c_clear = st.columns(2)
        with c_export:
            if st.button("CSV로 저장"):
                df_export = pd.DataFrame(st.session_state.inventory)
                st.download_button("다운로드", df_export.to_csv(index=False).encode('utf-8'), file_name="inventory.csv")
        with c_clear:
            if st.button("전체 재고 초기화"):
                if st.session_state.inventory:
                    st.session_state.last_deleted = st.session_state.inventory.copy()
                    st.session_state.inventory = []
                    st.success("초기화되었습니다.")
                    st.rerun()

# --- 3. AI 안전 스캐너 ---
elif menu == "📷 AI 안전 스캐너":
    st.title("🛡️ AI 안전 스캐너")
    img_file = st.camera_input("라벨 스캔")

    if img_file:
        original_img = Image.open(img_file)
        if st.button("🚀 분석 시작"):
            raw_text, cas_no = extract_cas_with_easyocr(original_img)

            if cas_no:
                info = CAS_DATABASE.get(cas_no, {"name": "미등록 물질", "hazard_level": "UNKNOWN", "msg": "정보 없음"})
                st.warning(f"### 분석 결과: {info['name']}")

                with st.container():
                    st.write("📦 **이 물품을 재고에 바로 추가하시겠습니까?**")
                    target_loc = st.selectbox("보관 위치 선택", available_locations, key="ai_loc")
                    if st.button("예, 추가합니다"):
                        st.session_state.inventory.append({
                            "id": int(time.time()), "name": info['name'], "category": "시약 및 화학물질",
                            "quantity": 1, "unit": "L", "cas": cas_no,
                            "hazard": info['hazard_level'], "status": "양호", "location": target_loc
                        })
                        st.success("재고에 등록되었습니다.")
            else:
                st.error("CAS 번호를 찾을 수 없습니다.")

# --- 4. 실험 레시피 ---
elif menu == "🔬 실험 레시피":
    st.title("🔬 실험 레시피 — 레시피 기반 재료 제안 및 위험 예측")

    recipe_names = list(RECIPES.keys())
    chosen = st.selectbox("레시피 선택", ["직접 입력/검색"] + recipe_names)

    if chosen == "직접 입력/검색":
        q = st.text_input("실험명 또는 키워드 입력")
        if q:
            matches = [name for name in recipe_names if q.lower() in name.lower()]
            if matches:
                chosen = st.selectbox("검색 결과", matches)

    if chosen and chosen != "직접 입력/검색":
        recipe = RECIPES[chosen]
        st.subheader(f"{chosen}")
        st.write(recipe.get('description', ''))

        reagents = recipe['reagents']

        # 재고 체크 및 위험 표시
        st.markdown("**필요 시약 목록**")
        missing = []
        overall_hazards = []

        for r in reagents:
            info = get_hazard_info_by_cas_or_name(r.get('cas', ''), r.get('name', ''))
            present = inventory_has_item(r)

            cols = st.columns([6, 2, 4])
            with cols[0]:
                st.write(f"• {r['name']} ({r.get('amount', '')})")
            with cols[1]:
                if present:
                    st.success("재고 있음")
                else:
                    st.error("부족")
                    missing.append(r)
            with cols[2]:
                st.info(f"{info.get('hazard_level')} — {info.get('msg')}")

            if info.get('hazard_level') in ['DANGER', 'WARNING']:
                overall_hazards.append(info.get('hazard_level'))

        # 상호작용 체크
        interaction_msgs = check_interactions(reagents)
        if interaction_msgs:
            st.warning("⚠️ 잠재적 위험 조합 감지:")
            for m in interaction_msgs:
                st.write(f"- {m}")

        # 종합 위험 레벨
        st.divider()
        if 'DANGER' in overall_hazards:
            st.error("종합 위험 수준: DANGER — 보호구 및 후드 사용 필요")
        elif 'WARNING' in overall_hazards:
            st.warning("종합 위험 수준: WARNING — 환기 및 주의 필요")
        else:
            st.success("종합 위험 수준: SAFE/UNKNOWN — MSDS 확인 권장")

        # 대체 시약 제안
        st.markdown("**대체 시약 제안**")
        subs = recipe.get('substitutes', {})
        if subs:
            for k, v in subs.items():
                st.write(f"• {k} → {v}")
        else:
            st.write("(등록된 대체 시약이 없습니다)")

        # 부족한 항목 추가 버튼
        if missing:
            if st.button("부족한 항목을 재고에 추가하기"):
                for r in missing:
                    st.session_state.inventory.append({
                        "id": int(time.time() * 1000) + int(np.random.randint(0, 100)),  # 고유 ID 생성
                        "name": r['name'],
                        "category": "시약 및 화학물질",
                        "quantity": 1,
                        "unit": "EA",
                        "cas": r.get('cas', '-'),
                        "hazard": get_hazard_info_by_cas_or_name(r.get('cas', ''), r.get('name', '')).get(
                            'hazard_level', 'UNKNOWN'),
                        "status": "양호",
                        "location": available_locations[0] if available_locations else "미지정"
                    })
                st.success("부족한 항목들을 재고에 추가했습니다.")
                st.rerun()

# --- 5. 실험실 지도 설정 ---
elif menu == "🗺️ 실험실 지도 설정":
    st.title("🗺️ 실험실 도면 및 구역 설정")
    st.info("사각형을 그려 실험실의 시약장이나 테이블 위치를 지정하세요.")

    col1, col2 = st.columns([2, 1])

    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(173, 216, 230, 0.4)",
            stroke_width=2,
            height=500, width=800,
            drawing_mode="rect",
            key=f"lab_canvas_{st.session_state.canvas_key}",
        )

    with col2:
        if st.button("🔄 지도 초기화"):
            st.session_state.lab_map = []
            st.session_state.canvas_key += 1
            st.rerun()

        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            objects = canvas_result.json_data["objects"]
            temp_map_data = []

            st.subheader("구역 이름 지정")
            for i, obj in enumerate(objects):
                with st.expander(f"구역 #{i + 1} 설정", expanded=True):
                    name = st.text_input(f"이름", value=f"Area {chr(65 + i)}", key=f"map_name_{i}")
                    temp_map_data.append({
                        "구역명": name,
                        "x": int(obj['left']),
                        "y": int(obj['top']),
                        "w": int(obj['width']),
                        "h": int(obj['height'])
                    })

            if st.button("💾 이 구조로 지도 저장", type="primary"):
                st.session_state.lab_map = temp_map_data
                st.success("지도가 저장되었습니다! 이제 재고 관리에서 이 구역들을 선택할 수 있습니다.")

# 1단계에서 했던 '웹에 게시' CSV 주소
READ_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRJu60BnrI181xEQbW7X1prv2AeLUCwXjDEF9Kxkgpk67nhAdWAmdwOh_kdx50wQwQfndQxRs3WmIHR/pubhtml"
# 방금 복사한 '웹 앱' 주소
WRITE_URL = "https://script.google.com/macros/library/d/12RFfbEXC2fUW2jeWp-2uHhZqxoJWUUeJM4f0FBOlPFRPBinpOw3oiqyV/4?pli=1&authuser=0"

# [수정 3] 재고 추가 로직 (AI 스캐너 메뉴 안의 '예, 추가합니다' 버튼 부분)
if st.button("예, 추가합니다"):
    new_item = {
        "id": int(time.time()),
        "name": info['name'],
        "category": "시약 및 화학물질",
        "quantity": 1,
        "unit": "L",
        "cas": cas_no,
        "hazard": info['hazard_level'],
        "status": "양호",
        "location": target_loc
    }

    # 구글 시트로 데이터 전송
    try:
        response = requests.post(WRITE_URL, data=json.dumps(new_item))
        if response.status_code == 200:
            st.success("✅ 구글 시트에 실시간 저장되었습니다!")
            st.rerun()  # 화면 새로고침해서 리스트 업데이트
        else:
            st.error("전송 실패. URL을 확인하세요.")
    except Exception as e:
        st.error(f"오류 발생: {e}")