config_name="./config/reader/base_gen_config.json"

# 기본값 설정
DO_EVAL=false

# Arguments 처리
for arg in "$@"
do
    case $arg in
        --do_eval)
        DO_EVAL=true
        shift
        ;;
    esac
done

# Evaluation 수행 여부 확인
if [ "$DO_EVAL" = true ]; then
    echo "Evaluation을 수행합니다..."
    # 여기에 evaluation 명령어 추가
    python src/modeling/mrc/train_mrc.py --config "$config_name" --do_train --do_eval
else
    echo "Evaluation을 건너뜁니다."
    python src/modeling/mrc/train_mrc.py --config "$config_name" --do_train
fi