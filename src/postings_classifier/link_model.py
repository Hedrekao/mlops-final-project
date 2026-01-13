import os
from typing import List, Optional

import typer
import wandb


app = typer.Typer()


def _resolve_entity() -> Optional[str]:
    return os.getenv("WANDB_ENTITY")


@app.command()
def link_model(artifact_path: str, aliases: Optional[List[str]] = None) -> None:
    """Stage a specific model artifact to the model registry.

    Args:
        artifact_path: Artifact reference in the form
            "entity/project/artifact_name:version".
        aliases: List of aliases to link the artifact with (e.g. ["staging"]).
    """
    if not artifact_path:
        typer.echo("No artifact path provided. Exiting.")
        raise typer.Exit(code=1)

    aliases = aliases or ["staging"]

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )

    try:
        # Expect artifact_path like "entity/project/artifact_name:version"
        _, _, artifact_name_version = artifact_path.rpartition("/")
        if not artifact_name_version:
            artifact_name_version = artifact_path
        artifact_name, _ = artifact_name_version.split(":", 1)
    except Exception:
        typer.echo("Invalid artifact_path format. Expected 'entity/project/artifact:version'.")
        raise typer.Exit(code=2)

    target_entity = _resolve_entity() or artifact_path.split("/", 1)[0]
    target_path = f"{target_entity}/model-registry/{artifact_name}"

    artifact = api.artifact(artifact_path)
    artifact.link(target_path=target_path, aliases=aliases)
    artifact.save()
    typer.echo(f"Artifact {artifact_path} linked to {target_path} with aliases={aliases}")


if __name__ == "__main__":
    app()
