## Saved models

### Trainsmall, no embeddings

The model is saved to the Wandb - to load it:
```
!pip install wandb
import wandb
wandb.login()

# Initialize Wandb
run = wandb.init(project="SLED-categorization", entity="tajak", name="testing-trained-model")

# Load the saved model
artifact = run.use_artifact('tajak/SLED-categorization/SLED-categorization-trainsmall-noembeddings-model:v0', type='model')
artifact_dir = artifact.download()

model = fasttext.load_model(f"{artifact_dir}/FastText-model-trainsmall-noembeddings.bin")
```

### Trainsmall, embeddings

The model is saved to the Wandb - to load it:
```
!pip install wandb
import wandb
wandb.login()

# Initialize Wandb
run = wandb.init(project="SLED-categorization", entity="tajak", name="testing-trained-model")

# Load the saved model
artifact = run.use_artifact('tajak/SLED-categorization/SLED-categorization-trainsmall-embeddings-model:v0', type='model')
artifact_dir = artifact.download()

model = fasttext.load_model(f"{artifact_dir}/FastText-model-trainsmall-embeddings.bin")
```

### Trainlarge, no embeddings

The model is saved to the Wandb - to load it:
```
!pip install wandb
import wandb
wandb.login()

# Initialize Wandb
run = wandb.init(project="SLED-categorization", entity="tajak", name="testing-trained-model")

# Load the saved model
artifact = run.use_artifact('tajak/SLED-categorization/SLED-categorization-trainlarge-noembeddings-model:v0', type='model')
artifact_dir = artifact.download()

model = fasttext.load_model(f"{artifact_dir}/FastText-model-trainlarge-noembeddings.bin")
```

### Trainlarge, embeddings

The model is saved to the Wandb - to load it:
```
!pip install wandb
import wandb
wandb.login()

# Initialize Wandb
run = wandb.init(project="SLED-categorization", entity="tajak", name="testing-trained-model")

# Load the saved model
artifact = run.use_artifact('tajak/SLED-categorization/SLED-categorization-trainlarge-embeddings-model:v0', type='model')
artifact_dir = artifact.download()

model = fasttext.load_model(f"{artifact_dir}/FastText-model-trainlarge-embeddings.bin")
```