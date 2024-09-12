model_list=(
    "llama2-13b-chat"
    "llama2-13b"
    "llama2-7b-chat"
    "llama2-7b"
    "qwen2-7b-instruct"
    "baichuan2-13b-chat"
    "baichuan2-7b-chat"
    "glm-4-9b-chat"
    "gpt-3.5-turbo-1106"
    "gpt-4o"
)
input_dir="/data/"
output_dir="/result/"

# evaluate transparency
evaluation_type="transparency"
for model_name in "${model_list[@]}"; do
    echo "Evaluating $model_name"
    python evaluation.py \
        --model_name $model_name \
        --evaluation_type $evaluation_type \
        --input_dir $input_dir \
        --output_dir $output_dir
done

# evaluate factuality
evaluation_type="factuality"
for model_name in "${model_list[@]}"; do
    echo "Evaluating $model_name"
    python evaluation.py \
        --model_name $model_name \
        --evaluation_type $evaluation_type \
        --input_dir $input_dir \
        --output_dir $output_dir
done

# evaluate robustness
evaluation_type="robustness"
for doc_num in 10; do
    for model_name in "${model_list[@]}"; do
        echo "Evaluating $model_name"
        python evaluation.py \
            --model_name $model_name \
            --evaluation_type $evaluation_type \
            --doc_num $doc_num \
            --save_note "$doc_num-docs" \
            --input_dir $input_dir \
            --output_dir $output_dir
    done
done

# evaluate accountability
evaluation_type="accountability"
for doc_num in 10; do
    for model_name in "${model_list[@]}"; do
        echo "Evaluating $model_name"
        python evaluation.py \
            --model_name $model_name \
            --evaluation_type $evaluation_type \
            --doc_num $doc_num \
            --save_note "$doc_num-docs" \
            --input_dir $input_dir \
            --output_dir $output_dir
    done
done


# evaluate privacy
evaluation_type="privacy"
for model_name in "${model_list[@]}"; do
    echo "Evaluating $model_name"
    python evaluation.py \
        --model_name $model_name \
        --evaluation_type $evaluation_type \
        --input_dir $input_dir \
        --output_dir $output_dir
done

# evaluate fairness
evaluation_type="fairness"
for model_name in "${model_list[@]}"; do
    echo "Evaluating $model_name"
    python evaluation.py \
        --model_name $model_name \
        --evaluation_type $evaluation_type \
        --input_dir $input_dir \
        --output_dir $output_dir
done