{
    "train_data_path": "projects/data/event/train_dataset.data",
    "validation_data_path": "projects/data/event/val_dataset.data",

    "dataset_reader": {

        "type": "TriggerDetectionDatasetReader",

        "tokenizer": {
            "type": "character"
        },

        "token_indexer": {
            "type": "single_id",
            "namespace": "tokens"
        },

        "lazy": true
    },

    "model": {
        "type": "TriggerDetectionModel",


        "sentence_embedder" : {
            "type": "basic",
            "token_embedders": {

                "character": {
                    "type": "embedding",
                    "embedding_dim": 50
                }
            }

        },

        "encoder": {
            "type": "lstm",
            "input_size": 50,
            "hidden_size": 100,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": true
        },

        "label_encoding": "BIO",
        "label_namespace": "labels", # 与label_field的name_space是一样的
        "crf": true # 暂时关闭crf
    },

    "iterator": {
        "type": "basic",
        "batch_size": 32
    },

    "trainer": {
        "validation_metric": "+accuracy",
        "optimizer": "adam",
        "num_epochs": 32,
        "patience": 10,
        "cuda_device": -1
    }

}