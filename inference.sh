# 기본 설정
config_name="./ODQA/config/base_config_sparse_v006.json"
pred_dir="./ODQA/predictions"
DO_VALID=false
DO_PREDICT=false
DO_EVAL=false
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
            if [ "$DO_PREDICT" = false ]; then
                echo "에러: --do_valid는 --do_predict와 함께만 사용할 수 있습니다."
                exit 1
            fi
            DO_VALID=true
            shift
            ;;
        --do_predict)
            if [ "$DO_EVAL" = true ]; then
                echo "에러: --do_predict와 --do_eval은 동시에 사용할 수 없습니다."
                exit 1
            fi
            DO_PREDICT=true
            shift
            ;;
        --do_eval)
            if [ "$DO_PREDICT" = true ]; then
                echo "에러: --do_predict와 --do_eval은 동시에 사용할 수 없습니다."
                exit 1
            fi
            DO_EVAL=true
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

# --do_predict와 --do_valid가 연관된지 체크
if [ "$DO_PREDICT" = true ] && [ "$DO_VALID" = false ]; then
    echo "에러: --do_predict를 사용할 때는 --do_valid가 필수입니다."
    exit 1
fi

# 실행 분기
if [ "$DO_PREDICT" = true ]; then
    echo "Prediction을 수행합니다..."
    python ./ODQA/inference.py \
        --config "$config_name" \
        --pred_dir "$pred_dir" \
        --do_predict \
        --do_valid \
        --qa "$QA_MODEL" \
        --retrieval "$RETRIEVAL_MODEL"
elif [ "$DO_EVAL" = true ]; then
    echo "Evaluation을 수행합니다..."
    python ./ODQA/inference.py \
        --config "$config_name" \
        --pred_dir "$pred_dir" \
        --do_eval \
        --qa "$QA_MODEL" \
        --retrieval "$RETRIEVAL_MODEL"
else
    echo "에러: --do_predict 또는 --do_eval 중 하나를 선택해야 합니다."
    exit 1
fi

# bash inference.sh --qa ext --retrieval sparse --do_eval
