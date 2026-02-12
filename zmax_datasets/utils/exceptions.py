class ZMaxDatasetError(Exception): ...


class MissingDataTypeError(ZMaxDatasetError): ...


class SleepScoringReadError(ZMaxDatasetError): ...


class SleepScoringFileNotFoundError(ZMaxDatasetError): ...


class SleepScoringFileNotSet(ZMaxDatasetError): ...


class MultipleSleepScoringFilesFoundError(ZMaxDatasetError): ...


class InvalidZMaxDataTypeError(ZMaxDatasetError): ...


class NoFeaturesExtractedError(ZMaxDatasetError): ...


class ChannelDurationMismatchError(ZMaxDatasetError): ...


class RawDataReadError(ZMaxDatasetError): ...


class HypnogramMismatchError(ZMaxDatasetError):
    def __init__(self, features_length: int, hypnogram_length: int):
        self.message = (
            "Features and hypnogram have different lengths:"
            f" {features_length} and {hypnogram_length}"
        )
        super().__init__(self.message)


class RecordingNotFoundError(ZMaxDatasetError): ...


class SampleRateNotFoundError(ZMaxDatasetError): ...




class InvalidFilterWindowError(ZMaxDatasetError):
    def __init__(self, window_size: int):
        message = f"Median filter window must be at least one sample, got {window_size}."
        super().__init__(message)


class InvalidChannelCountError(ZMaxDatasetError):
    def __init__(self, n_channels: int):
        message = f"Expected exactly one channel, got {n_channels}."
        super().__init__(message)


class MissingChannelThresholdError(ZMaxDatasetError):
    def __init__(self, channel: str):
        message = f"No thresholds found for channel {channel!r}."
        super().__init__(message)


class MissingFeatureThresholdError(ZMaxDatasetError):
    def __init__(self, feature: str, channel: str):
        message = f"Missing threshold for feature {feature!r} (channel {channel!r})."
        super().__init__(message)


class NoValidEEGChannelsError(ZMaxDatasetError):
    def __init__(self, *, expected: tuple[str, ...] | None = None, available: list[str] | None = None):
        msg = "No valid EEG channels found."
        if expected is not None:
            msg += f" Expected at least one of {expected!r}."
        if available is not None:
            msg += f" Available channels: {available}"
        super().__init__(msg)



class MissingColumnError(ZMaxDatasetError):
    def __init__(self, column: str):
        message = f"Missing required column: {column}"
        super().__init__(message)


class EmptyDataFrameError(ZMaxDatasetError):
    def __init__(self):
        message = "Input dataframe is empty."
        super().__init__(message)


class ChannelNotFoundError(ZMaxDatasetError):
    def __init__(self, channel: str, *, available: list[str] | None = None, context: str | None = None):
        msg = f"Channel {channel!r} not found."
        if context:
            msg += f" ({context})"
        if available is not None:
            msg += f" Available channels: {available}"
        super().__init__(msg)

        
class MultipleChannelsError(ZMaxDatasetError):
    def __init__(self, n_channels: int):
        message = (
            f"Expected exactly one channel, but epochs contains {n_channels}. "
            "Provide channel_name explicitly."
        )
        super().__init__(message)
        
class EpochCountMismatchError(ZMaxDatasetError):
    def __init__(self, left_count: int, right_count: int):
        super().__init__(f"Epoch count mismatch: L={left_count} R={right_count}")


class SampleRateMismatchError(ZMaxDatasetError):
    def __init__(self, expected: float, got: float):
        super().__init__(f"Data must have sample rate {expected}, got {got}")

class InvalidEpochDurationError(ZMaxDatasetError):
    def __init__(self, epoch_duration: float):
        super().__init__(f"epoch_duration must be positive, got {epoch_duration}")

class EpochLengthTooSmallError(ZMaxDatasetError):
    def __init__(self, epoch_duration: float, sample_rate: float):
        super().__init__(f"epoch_duration={epoch_duration} too small for sample_rate={sample_rate}")

class IncompleteEpochsError(ZMaxDatasetError):
    def __init__(self, data_length: int, epoch_length: int):
        super().__init__(f"Data length {data_length} is not a multiple of epoch_length {epoch_length}.")

class NoSamplesError(ZMaxDatasetError):
    def __init__(self, data_length: int, epoch_length: int):
        super().__init__(f"No complete epochs found: data_length={data_length}, epoch_length={epoch_length}")


