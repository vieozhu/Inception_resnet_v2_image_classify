rm -rf savedmodel/*
python ~/software/script/inception_resnet_v2_saved_model.py --checkpoint_dir=model --output_dir=savedmodel --label_file=label.txt --num_top_class=1 --num_class=4
