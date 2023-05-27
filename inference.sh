for num in {8..8} 
do
    input_folder=test_${num}

    # weight_name=step3_gan_3LBOs_4_net_g_180000
    weight_name=AnimeSR_v2

    # expname=animesr_v2
    expname=$weight_name

    echo '*******************Inference Info*******************'
    echo 'input_folder:     '$input_folder
    echo 'weight_name:      '$weight_name
    echo '******************Inference Starting****************'
    CUDA_VISIBLE_DEVICES=6 python scripts/inference_animesr_frames.py -i inputs/$input_folder -n $weight_name --expname $expname --save_video_too --fps 20
    echo '******************Inference Finished*****************'
done