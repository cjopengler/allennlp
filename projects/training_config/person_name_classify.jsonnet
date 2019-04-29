{
    "train_data_path": "projects/data/person_name_classify/person_name.train.data",
    "validation_data_path": "projects/data/person_name_classify/person_name.val.data",
    "dataset_reader": {
        "type": "PersonNameDatasetReader",
        "tokenizer": {
            "type": "character",
            "lowercase_characters": true
        }
    },
    "model": {
        "type": "PersonNameClassifier",

        "text_embedder": {
            "token_embedders": {
                "characters": {
                    "type": "embedding",

                    "embedding_dim": 100,
                    "trainable": true
                }
            }
        },

        "encoder": {
            "type": "gru",
            "bidirectional": true,
            "input_size": 100,
            "hidden_size": 100,
            "num_layers": 1,
            "dropout": 0.2

        },

        "feed_forward": {
            "input_dim": 200,
            "num_layers": 2,
            "hidden_dims": [64, 2],
            "activations": ["sigmoid", "linear"],
            "dropout": [0.2, 0.2]
        }
    },

    "iterator": {
        "type": "bucket",
        "sorting_keys": [["name", "num_tokens"]],
        "padding_noise": 0.0,
        "batch_size": 32
    },

    "trainer": {
        "num_epochs": 32,
        "cuda_device": -1,
        "grad_clipping": 5.0,
        "validation_metric": "+accuracy",
        "optimizer": {
          "type": "adam"
        }
    }
}