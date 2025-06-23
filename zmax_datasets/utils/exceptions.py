class ZMaxDatasetError(Exception): ...


class MissingDataTypeError(ZMaxDatasetError): ...


class SleepScoringReadError(ZMaxDatasetError): ...


class SleepScoringFileNotFoundError(ZMaxDatasetError): ...


class SleepScoringFileNotSet(ZMaxDatasetError): ...


class MultipleSleepScoringFilesFoundError(ZMaxDatasetError): ...


class InvalidZMaxDataTypeError(ZMaxDatasetError): ...


class NoFeaturesExtractedError(ZMaxDatasetError): ...


class ChannelLengthMismatchError(ZMaxDatasetError): ...


class HypnogramMismatchError(ZMaxDatasetError):
    def __init__(self, features_length: int, hypnogram_length: int):
        self.message = (
            "Features and hypnogram have different lengths:"
            f" {features_length} and {hypnogram_length}"
        )
        super().__init__(self.message)
