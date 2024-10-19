config_name="..reader/extractive/config/reader/base_config.json"
pred_dir="./predictions"
# 기본값 설정
DO_VALID=false

# Arguments 처리
for arg in "$@"
do
    case $arg in
        --do_valid)
        DO_VALID=true
        shift
        ;;
    esac
done


# Validation 수행 여부 확인
if [ "$DO_VALID" = true ]; then
    echo "Validation Prediction을 수행합니다..."
    python src/inference_2.py --config "$config_name" --pred_dir "$pred_dir" --do_predict --do_eval --do_valid
else
    echo "Validation Prediction을 건너뜁니다."
    python src/inference_2.py --config "$config_name" --pred_dir "$pred_dir" --do_predict --do_eval
fi

