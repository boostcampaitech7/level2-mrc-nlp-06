# 기본 설정
config_name="./ODQA/config/base_config.json"
pred_dir="./ODQA/predictions"
DO_VALID=false
QA_MODEL=""
RETRIEVAL_MODEL=""

# 명령줄 인자 처리
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --qa)
            QA_MODEL="$2"
            shift 2
            ;;
        --retrieval)
            RETRIEVAL_MODEL="$2"
            shift 2
            ;;
        --do_valid)
            DO_VALID=true
            shift
            ;;
        *)
            echo "알 수 없는 인자: $1"
            exit 1
            ;;
    esac
done

# 필수 인자 체크
if [ -z "$QA_MODEL" ] || [ -z "$RETRIEVAL_MODEL" ]; then
    echo "에러: --qa와 --retrieval 인자는 필수입니다."
    exit 1
fi

# Validation 수행 여부에 따른 실행 분기
if [ "$DO_VALID" = true ]; then
    echo "Validation Prediction을 수행합니다..."
    python ./ODQA/inference.py \
        --config "$config_name" \
        --pred_dir "$pred_dir" \
        --do_valid \
        --qa "$QA_MODEL" \
        --retrieval "$RETRIEVAL_MODEL"
else
    echo "Validation Prediction을 건너뜁니다."
    python ./ODQA/inference.py \
        --config "$config_name" \
        --pred_dir "$pred_dir" \
        --qa "$QA_MODEL" \
        --retrieval "$RETRIEVAL_MODEL"
fi