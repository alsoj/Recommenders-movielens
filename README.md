# Recommenders-movielens
## movielens dataset을 활용한 추천 시스템 구현

사용 알고리즘 : 협업 필터링(Collaborative Filtering), Softmax, Wide & Deep

recommender_CF&Softmax.ipynb : 협업 필터링(Collaborative Filtering), Softmax
recommender_wide&deep.ipynb : Wide & Deep

recommender_wide&deep.ipynb 중

모델 생성 function
def build_model(
    model_dir=MODEL_DIR,
    wide_columns=(),
    deep_columns=(),
    linear_optimizer='Ftrl',
    dnn_optimizer='Adagrad',
    dnn_hidden_units=(128, 128),
    dnn_dropout=0.0,
    dnn_batch_norm=True,
    log_every_n_iter=1000,
    save_checkpoints_steps=10000
):


Wide Column만 있을 때
model = tf.estimator.LinearRegressor

Deep Column만 있을 때
model = tf.estimator.DNNRegressor

Wide Column과 Deep Column이 모두 있을 때
model = tf.estimator.DNNLinearCombinedRegressor
