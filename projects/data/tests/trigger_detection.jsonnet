{
    "train_data_path": "data/projects/tests/training.data",
    "validation_data_path": "data/projects/tests/training.data",

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
        }
    },

    "iterator": {
        "type": "basic",
        "batch_size": 32
    },

    "trainer": {
        "validation_metric": "+accuracy",
        "optimizer": "adam",
        "num_epochs": 10,
        "patience": 10,
        "cuda_device": -1
    }

}