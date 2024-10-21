config_name="./reader/extractive/config/base_config.json"

# 기본값 설정
DO_EVAL=false
TYPE=""

# Arguments 처리
for arg in "$@"
do
    case $arg in
        --do_eval)
        DO_EVAL=true
        shift
        ;;
        --type=*)
        TYPE="${arg#*=}"
        shift
        ;;
    esac
done

# Evaluation 수행 여부와 타입 확인
if [ "$TYPE" = "abs" ]; then
    echo "Abstract 타입으로 실행합니다..."
    if [ "$DO_EVAL" = true ]; then
        echo "Evaluation을 수행합니다..."
        python reader/abstractive/src/train_abs.py --config "$config_name" --do_train --do_eval
    else
        echo "Evaluation을 건너뜁니다."
        python reader/abstractive/src/train_abs.py --config "$config_name" --do_train
    fi
elif [ "$TYPE" = "ext" ]; then
    echo "Extract 타입으로 실행합니다..."
    if [ "$DO_EVAL" = true ]; then
        echo "Evaluation을 수행합니다..."
        python reader/extractive/src/train_ext.py --config "$config_name" --do_train --do_eval
    else
        echo "Evaluation을 건너뜁니다."
        python reader/extractive/src/train_ext.py --config "$config_name" --do_train
    fi
else
    echo "올바른 --type 인자를 제공해 주세요. (abs 또는 ext)"
    exit 1
fi

# bash reader/train_mrc.sh --type=ext --do_eval