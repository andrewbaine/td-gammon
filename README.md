

Build docker images
```
docker build -t td-gammon .
docker build -t gnubg ./gnubg
```

Build Move Tensors
```
./scripts/build_move_tensors.sh
```

Train a neural network
```
./scripts/train.sh -o 4 -g 400000 -m my-awesome-model
```

Evaluate a model
```
./scripts/train.sh -o 4 -g 400000 -m var/models/my-awesome-model/model.00200000.pt
```


source .venv/bin/activate

pytest


Performance Testing checklist

- maybe do with torch.no_grad()?
- do performatnce profiling?
