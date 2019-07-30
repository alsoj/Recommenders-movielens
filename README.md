# 이 REPOSITORY는
Movie-Lens Dataset을 활용해 추천시스템을 구현한 프로젝트입니다.  
사용 알고리즘 : 협업 필터링(Collaborative Filtering), Softmax, Wide & Deep  
recommender_CF&Softmax.ipynb : 협업 필터링(Collaborative Filtering), Softmax  
recommender_wide&deep.ipynb : Wide & Deep  

## About Wide & Deep Model
###Wide & Deep Model을 이해하기 위한 참고 자료  
[구글 Wide & Deep 논문] https://arxiv.org/abs/1606.07792  
[PR12 논문 리뷰] https://youtu.be/hKoJPqWLrI4  
[논문 한글 설명] https://yamalab.tistory.com/101  
[Wide & Deep 튜토리얼] https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/wide_and_deep/  

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
