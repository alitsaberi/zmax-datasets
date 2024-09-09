class ZMaxDatasetError(Exception): ...


class MissingDataTypesError(ZMaxDatasetError):
    def __init__(self, missing_data_types: list[str]):
        self.missing_data_types = missing_data_types
        self.message = f"Missing data types: {', '.join(missing_data_types)}"
        super().__init__(self.message)


class SleepScoringReadError(ZMaxDatasetError): ...


class SleepScoringFileNotFoundError(ZMaxDatasetError): ...


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
