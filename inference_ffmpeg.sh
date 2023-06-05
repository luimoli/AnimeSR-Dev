# folders=('A' 'B' 'C' 'D' 'E')
# result_folder='animesr_v2_s2_r0_5'

folders=('A' 'B' 'D' 'E')
result_folder='animesr_v2_s2'

for folder in "${folders[@]}"
do
    echo '*******************Info*******************'
    echo 'result_folder:    '$result_folder
    echo 'input_folder:     '$folder
    echo '******************encode Starting****************'
    ffmpeg -i results/$result_folder/frames/$folder/%5d.png -c:v libx264 -r 25 -crf 0 \
		-color_range tv \
		-colorspace bt709 \
		-color_primaries bt709 \
		-color_trc bt709 \
		-vf "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=pc:out_range=tv" \
		results/$result_folder/videos/$folder.mp4
    echo '******************encode Finished*****************'
done

