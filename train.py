from ultralytics import YOLO
from utils.model_handler import best_model

if __name__ == '__main__':
    # TODO: Move this value maybe to env or like a arguments of script
    experiment = 'version'
    # TODO: Maybe move all dataset to google drive and on start download it
    data_path = 'data.yaml'
    n_epochs = 1
    bs = 8
    n_workers = bs
    gpu_id = 0
    verbose = True
    rng = 0
    validate = True
    patience = 0
    project = './versions'

    model = YOLO(best_model())

    results = model.train(
        # device gpu
        device="0",
        data=data_path,
        epochs=n_epochs,
        batch=bs,
        verbose=verbose,
        seed=rng,
        val=validate,
        project=project,
        name=experiment,
        workers=n_workers,
        patience=patience
    )
