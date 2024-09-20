from zmax_datasets.exports.enums import ErrorHandling, ExistingFileHandling
from zmax_datasets.exports.yasa import YasaExportStrategy


def test_yasa_export_strategy_initialization():
    eeg_channel = "EEG R"
    eog_channel = "EEG L"
    sampling_frequency = 100
    test_split_size = 0.2

    strategy = YasaExportStrategy(
        eeg_channel=eeg_channel,
        eog_channel=eog_channel,
        sampling_frequency=sampling_frequency,
        test_split_size=test_split_size,
        existing_file_handling=ExistingFileHandling.RAISE_ERROR,
        error_handling=ErrorHandling.RAISE,
    )

    assert strategy.eeg_channel == eeg_channel
    assert strategy.eog_channel == eog_channel
    assert strategy.sampling_frequency == sampling_frequency
    assert strategy.test_split_size == test_split_size
    assert strategy.existing_file_handling == ExistingFileHandling.RAISE_ERROR
    assert strategy.error_handling == ErrorHandling.RAISE
