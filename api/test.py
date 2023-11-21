import model_utils as model_utils

# model, tokenizer = model_utils.load()
# labels = model_utils.label_extractor(path="labels.csv")
# le = model_utils.label_encoder()

test1 = 'defaut de com ipc'
test2 = 'blabla hmi cassé'
# max_seq_length = 32

# preds = model_utils.prediction(model, max_seq_length, tokenizer, [test1, test2])

# print(preds)


print(model_utils.predict_pipeline([test1,test2]))