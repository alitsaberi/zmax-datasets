from loguru import logger

from zmax_datasets.processing.eeg_usability import (
    get_usability_scores,
    load_model,
)

from zmax_datasets.transforms.base import Transform
from zmax_datasets.utils.data import Data

from zmax_datasets.processing.eeg_artifact_detect import (
    run_full_zmax_artifact_pipeline_from_data,
)



class EEGUsability(Transform):
    def __init__(self, model_version: str = "default"):
        self._model_version = model_version
        self._model = load_model(model_version)

    def __call__(self, data: Data) -> Data:
        """
        Args:
            data (Data): Input data containing EEG left, EEG right, movement.

        Returns:
            Data: EEG usability scores
        """
        if data.n_channels != 3:
            raise ValueError(
                f"Expected 3 channels (EEG left, EEG right, movement), "
                f"got {data.n_channels}"
            )

        usability_scores, _, _ = get_usability_scores(
            data, self._model, *data.channel_names
        )

        logger.info(
            "EEG left artifact count:"
            f" {usability_scores[:, 0].array.sum()}/{usability_scores.length}%"
        )
        logger.info(
            "EEG right artifact count:"
            f" {usability_scores[:, 1].array.sum()}/{usability_scores.length}%"
        )

        return usability_scores
    
    
class EEGArtifactDetection(Transform):
    """
    Rule-based EEG artifact & usability transform using the full
    run_full_zmax_artifact_pipeline_from_data pipeline.

    Output:
        Data with one sample per 30-s epoch:
        - channel 0: continuous usability score (0–1)
        - channel 1: binary usability_score (1 if usability > threshold)
        - channel 2: artifact flag (1 if n_bad_channels > 0)
    """

    def __init__(
        self,
        left_label: str = "EEG_L",
        right_label: str = "EEG_R",
        epoch_length: float = 30.0,
        usability_threshold: float = 0.4,
        plot: bool = False,
    ):
        self.left_label = left_label
        self.right_label = right_label
        self.epoch_length = epoch_length
        self.usability_threshold = usability_threshold
        self.plot = plot

    def __call__(self, data: Data) -> Data:
        # Run your big pipeline (no plotting in production by default)
        out = run_full_zmax_artifact_pipeline_from_data(
            data=data,
            left_label=self.left_label,
            right_label=self.right_label,
            sf=float(data.sample_rate),
            plot_eeg=self.plot,
        )

        usability_df = out["usability_df"].copy()

        # --- derive binary usability + artifact flags ---
        usability_df["usability_score"] = (
            usability_df["usability"] > self.usability_threshold
        ).astype(int)

        usability_df["artifact"] = (usability_df["n_bad_chan"] > 0).astype(int)

        n_epochs = len(usability_df)
        n_artifacts = int(usability_df["artifact"].sum())
        logger.info(
            "Rule-based artifact epochs (any channel bad): "
            f"{n_artifacts}/{n_epochs}"
        )

        # Turn into Data object: (n_epochs, 3)
        arr = usability_df[["usability", "usability_score", "artifact"]].to_numpy()

        return Data(
            array=arr,
            sample_rate=1.0 / self.epoch_length,  # 1 sample per 30 s
            channel_names=[
                "usability_rb",
                "usability_rb_bin",
                "artifact_rb",
            ],
        )

