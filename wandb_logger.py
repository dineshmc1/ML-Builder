import os
from config import USE_WANDB, WANDB_PROJECT, WANDB_ENTITY

_run = None

def init_run(run_name: str, config: dict):
    global _run
    if not USE_WANDB:
        return
    import wandb
    os.environ["WANDB_SILENT"] = "true"
    _run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=config,
        reinit=True
    )

def log(metrics: dict, step: int = None):
    if not USE_WANDB or _run is None:
        return
    import wandb
    wandb.log(metrics, step=step)

def log_image(key: str, figure):
    if not USE_WANDB or _run is None:
        return
    import wandb
    wandb.log({key: wandb.Image(figure)})

def log_table(key: str, dataframe):
    if not USE_WANDB or _run is None:
        return
    import wandb
    wandb.log({key: wandb.Table(dataframe=dataframe)})

def log_artifact(filepath: str, artifact_type: str, name: str):
    if not USE_WANDB or _run is None:
        return
    import wandb
    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(filepath)
    _run.log_artifact(artifact)

def finish():
    if not USE_WANDB or _run is None:
        return
    import wandb
    wandb.finish()

def alert(title: str, text: str):
    if not USE_WANDB or _run is None:
        return
    import wandb
    wandb.alert(title=title, text=text)
