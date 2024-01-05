import argparse
from ultralytics import YOLO
from utils.model_handler import best_model

parser = argparse.ArgumentParser(
    prog="Train Model"
)

parser.add_argument('--model_name', default="version")
parser.add_argument('--out_path', default='./versions')
parser.add_argument('--epochs', default=10)
parser.add_argument('--gpu_id', default=0)

if __name__ == '__main__':
    # TODO: Maybe move all dataset to google drive and on start download it
    data_path = 'data.yaml'
    bs = 8
    n_workers = bs
    verbose = True
    rng = 0
    validate = True
    patience = 0
    args = parser.parse_args()

    model = YOLO(best_model(args.out_path))

    results = model.train(
        # device gpu
        device=args.gpu_id,
        data=data_path,
        epochs=int(args.epochs),
        batch=bs,
        verbose=verbose,
        seed=rng,
        val=validate,
        project=args.out_path,
        name=args.model_name,
        workers=n_workers,
        patience=patience
    )
