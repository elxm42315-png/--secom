import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # 결측치 처리를 위해 추가
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

# 모델 및 스케일러 파일 경로
MODEL_PATH = 'xgboost_semi_model.pkl'
TARGET_COL = 'label' # 대상 컬럼 이름

# Streamlit 애플리케이션 제목 설정
st.set_page_config(
    page_title="반도체 공정 불량 예측 대시보드 (XGBoost)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. 데이터 로드 및 전처리 함수 ---
@st.cache_data
def load_data():
    """데이터셋을 로드하고 필요한 기본 전처리를 수행합니다."""
    # 사용자가 업로드한 '반도체_데이터.csv' 파일을 로드
    df = pd.read_csv("반도체_데이터.csv")
    
    # 컬럼 이름의 공백/불필요한 문자 제거
    df.columns = df.columns.str.strip()
    
    # Feature 컬럼 목록 (label 제외)
    feature_cols = df.columns.tolist()
    if TARGET_COL in feature_cols:
        feature_cols.remove(TARGET_COL)
    else:
        st.error(f"데이터셋에 타겟 컬럼 '{TARGET_COL}'이 없습니다. 컬럼명을 확인해 주세요.")
        return pd.DataFrame(), [], TARGET_COL
    
    # Target 컬럼을 정수형으로 변환
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    
    return df, feature_cols, TARGET_COL

# 데이터 로드
df, all_feature_cols, TARGET_COL = load_data()

# --- 2. 모델 학습 및 저장 함수 ---
@st.cache_resource
def train_and_save_model(df_model, features, target_col, model_path):
    """XGBoost 모델을 학습하고 저장합니다."""
    
    X = df_model[features]
    y = df_model[target_col]

    # 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 결측치 처리 (Median Imputer 사용)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # XGBoost 모델 학습 (불균형 데이터 처리를 위해 scale_pos_weight 사용)
    # y=0 (Pass) / y=1 (Fail)
    ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=ratio, 
        n_estimators=100
    )
    model.fit(X_train_scaled, y_train)
    
    # 모델 저장에 사용된 특성 목록 저장
    model_features = X_train.columns.tolist()
    
    # 모델, 스케일러, 임퓨터, 특성 목록 저장
    joblib.dump((model, scaler, imputer, model_features), model_path)
    
    # 성능 평가
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    st.success(f"✅ XGBoost 모델 학습 완료 및 저장! (Test Accuracy: {accuracy:.4f}, Test AUC: {roc_auc:.4f})")
    
    return model, scaler, imputer, model_features

# --- 3. 모델 학습/로드 로직 ---
st.sidebar.title("⚙️ 모델 학습 및 정보")
if not df.empty:
    st.sidebar.info("새로운 데이터가 로드되어 모델을 학습합니다.")
    try:
        model, scaler, imputer, feature_subset = train_and_save_model(
            df.copy(), all_feature_cols, TARGET_COL, MODEL_PATH
        )
    except Exception as e:
        st.error(f"모델 학습 중 오류가 발생했습니다: {e}")
        model, scaler, imputer, feature_subset = None, None, None, []
else:
    model, scaler, imputer, feature_subset = None, None, None, []
    st.sidebar.error("데이터 로드 실패 또는 데이터가 비어 있습니다.")


# --- 4. 대시보드 구성 ---

st.title("🏭 반도체 공정 불량률 예측 대시보드")
st.markdown("이 대시보드는 **반도체\_데이터.csv**를 기반으로 **XGBoost 모델**을 사용하여 불량(label=1)을 예측합니다.")

st.markdown("---")

## 💡 XGBoost 불량 예측 시연

st.subheader("모델 예측 시연")

if model is not None:
    # Target 0(양품)과 1(불량)인 샘플의 인덱스 목록
    pass_indices = df[df[TARGET_COL] == 0].index.tolist()
    fail_indices = df[df[TARGET_COL] == 1].index.tolist()
    
    prediction_mode = st.radio("예측 모드 선택", ["랜덤 Pass (양품) 샘플 예측", "랜덤 Fail (불량) 샘플 예측"])
    
    # NameError 방지를 위해 sample_data를 미리 None으로 초기화
    sample_data = None 
    sample_index = None

    if prediction_mode == "랜덤 Pass (양품) 샘플 예측" and pass_indices:
        sample_index = np.random.choice(pass_indices)
        sample_data = df.loc[sample_index]
        actual_label_text = "Pass (양품)"
    elif prediction_mode == "랜덤 Fail (불량) 샘플 예측" and fail_indices:
        sample_index = np.random.choice(fail_indices)
        sample_data = df.loc[sample_index]
        actual_label_text = "Fail (불량)"
    else:
        st.warning("선택된 모드에 해당하는 샘플이 데이터에 부족합니다.")

    if sample_data is not None:
        
        # 예측 대상 데이터 준비 (모델 학습에 사용된 특성만 사용)
        X_sample = sample_data[feature_subset].values.reshape(1, -1)
        
        # 결측치 처리 및 스케일링
        X_sample_imputed = imputer.transform(X_sample)
        X_sample_scaled = scaler.transform(X_sample_imputed)
        
        # 예측 수행
        prediction_proba = model.predict_proba(X_sample_scaled)[:, 1][0]
        prediction_class = model.predict(X_sample_scaled)[0]
        
        col_pred_info, col_pred_result = st.columns([1, 1])
        
        with col_pred_info:
            st.info(f"**샘플 정보 (인덱스: {sample_index})**")
            
            # 🌟🌟🌟🌟🌟 TypeError 수정 부분 🌟🌟🌟🌟🌟
            # Series를 DataFrame으로 변환 후 전치하고 컬럼 이름 변경
            df_display = sample_data[feature_subset].head(10).to_frame().T
            df_display = df_display.rename(columns={sample_index: '값'})
            
            st.dataframe(df_display.T, use_container_width=True) # 다시 전치하여 세로로 표시
            # 🌟🌟🌟🌟🌟 수정 끝 🌟🌟🌟🌟🌟
            
        with col_pred_result:
            st.metric(
                label="예측된 불량(Fail, label=1) 확률",
                value=f"{prediction_proba * 100:.2f} %",
                delta_color="off"
            )
            
            if prediction_class == 1:
                st.error("🚨 예측 결과: **Fail (불량)**")
            else:
                st.success("✅ 예측 결과: **Pass (양품)**")
                
            st.caption(f"이 샘플의 실제 결과는 **{actual_label_text}**입니다.")
            
else:
    st.error("모델 학습에 실패하여 예측 기능을 사용할 수 없습니다. 데이터 상태를 확인해 주세요.")

st.markdown("---")

## 📊 데이터 시각화 및 분포 분석

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 생산 결과 분포")
    # Target (label) 빈도 계산
    target_counts = df[TARGET_COL].map({0: 'Pass (양품)', 1: 'Fail (불량)'}).value_counts().reset_index()
    target_counts.columns = ['Result', 'Count']
    
    fig_pie = px.pie(
        target_counts, 
        values='Count', 
        names='Result', 
        title='전체 샘플의 Pass/Fail 비율',
        color_discrete_sequence=['#636efa', '#ef553b']
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)
    
with col2:
    st.subheader("💡 특성 중요도")
    if model is not None:
        # 특성 중요도 시각화
        feature_importances = pd.Series(model.feature_importances_, index=feature_subset)
        top_10_features = feature_importances.nlargest(10)
        
        fig_importance = px.bar(
            top_10_features,
            orientation='h',
            title='상위 10개 특성 중요도 (Feature Importance)',
            labels={'value': '중요도', 'index': '특성'}
        )
        fig_importance.update_layout(showlegend=False, yaxis={'autorange':'reversed'})
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.info("모델이 학습되지 않아 특성 중요도를 표시할 수 없습니다.")

st.markdown("---")

## 📈 상세 특성 분포 비교

st.subheader("상세 특성 분포 비교")

if model is not None and feature_subset:
    # 중요도가 높은 특성들을 사이드바에서 선택할 수 있도록 제공
    feature_importances = pd.Series(model.feature_importances_, index=feature_subset)
    sorted_features = feature_importances.sort_values(ascending=False).index.tolist()
    
    hist_feature = st.selectbox(
        '분포를 볼 특성 선택 (중요도 순)', 
        sorted_features, 
        index=0
    )
    
    if hist_feature:
        st.info(f"선택된 특성: **{hist_feature}**")
        
        # Target 값을 다시 레이블로 매핑
        df_plot = df.copy()
        df_plot['Target_Label'] = df_plot[TARGET_COL].map({0: 'Pass (양품)', 1: 'Fail (불량)'})
        
        # 히스토그램 생성
        fig_hist = px.histogram(
            df_plot, 
            x=hist_feature, 
            color="Target_Label", 
            marginal="box",
            opacity=0.6,
            title=f'{hist_feature}의 Pass/Fail 분포',
            labels={'Target_Label': '생산 결과'}
        )
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("모델 학습에 필요한 데이터나 특성이 부족하여 분포 분석을 표시할 수 없습니다.")