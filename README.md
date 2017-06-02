# Google cloud commands

```sh
BUCKET_NAME=gs://${USER}_yt8m_train_bucket
```
## Training:
The parameters to be modified are from the "base_learning_rate" to "start_new_model":
* TRAIN_DIR: "yt8m_name_simulation" is the name of the simulation (and the folder where it will be saved). For 
another simulation, give another meaningful name, depending on what you are trying to do.
* base_learning_rate: Which learning rate to start with
* batch_size
* reg_lambda: controls the proportions between the two loss functions. If it is big, the classification loss will be 
bigger. It only trains correctly (at least what I've tried) when it starts being zero. Then you can change it
* percentage_negative: Percentage of negative samples (from 0 to 1)
* margin: related to the cosine loss. If margin is high (up to 1), the cosine loss does not punish negative embeddings 
with a cosine distance lower than margin.
* start_new_model: If you want to continue with the previous simulation (that has the same train_dir), it has to be 
False. If not, the simulation is overwritten.

Other useful modifications can be done in two files: 
* video_level_models.py, in the EmbeddingModel class. You can change anything you want, as long as the dimensions are 
correct (audio features always 128, video features always 1024). You can add more layers, change the size of the hidden 
layers...
* losses.py, in the CosineAndCrossEntropyLoss class. 

```sh
TRAIN_DIR=yt8m_name_simulation
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train --staging-bucket=$BUCKET_NAME \
--region=us-east1 --config=youtube-8m/cloudml-gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml-us-east1/1/video_level/train/train*.tfrecord' \
--train_dir=$BUCKET_NAME/${TRAIN_DIR} \
--base_learning_rate=0.0001 \
--batch_size=1024 \
--reg_lambda=0.0 \
--percentage_negative=0.6 \
--margin=0.2 \
--start_new_model=False \
--negative_sampling=True \
--model=EmbeddingModel \
--select_randomly=False \
--feature_names="mean_rgb, mean_audio" \
--feature_sizes="1024, 128" \
--num_readers=8 \
--image_server=False \
--label_loss="CosineAndCrossEntropyLoss" \
```

## Evaluation (Validation):
Meaningful parameters:
* JOB_TO_EVAL: it has to be the same as the TRAIN_DIR used in the training.
* batch_size: number of features among which the closest embedding will be looked for.
* hits: represents the "k" in Recall@k

If you want to create new evaluation metrics, the files you have to change are "eval" and
"eval_util". In the second one you can create a function such as "calculate_hit_at_k_embedding", to 
evaluate whatever you want, having the embeddings of all the samples of the batch as input.
Then you just have to see where the function "calculate_hit_at_k_embedding" is called, and do 
the same for your new function.

```sh
JOB_TO_EVAL=yt8m_name_simulation
BOARD=yt8m_name_simulation_board
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml-us-east1/1/video_level/validate/validate*.tfrecord' \
--hits=1 \
--batch_size=256 \
--model=EmbeddingModel \
--select_randomly=False \
--feature_names="mean_rgb, mean_audio" \
--feature_sizes="1024, 128" \
--label_loss="CosineAndCrossEntropyLoss" \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--run_once=True \
--board_dir=$BUCKET_NAME/${BOARD} 
``