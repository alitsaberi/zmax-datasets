from loguru import logger

from zmax_datasets.pipeline.configs import (
    PipelineConfig,
    PipelineStepConfig,
    TransformConfig,
)
from zmax_datasets.utils.data import Data


class Pipeline:
    """Simple linear pipeline executor"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        logger.info(f"Initialized pipeline: {config.name}")

    def execute(self, initial_data: dict[str, Data]) -> dict[str, Data]:
        """
        Execute pipeline steps using provided data.

        Args:
            initial_data: Dictionary mapping data type names to Data objects

        Returns:
            Dictionary mapping all data type names to Data objects
        """
        logger.info(
            f"Executing pipeline '{self.config.name}'"
            f" with initial data types: {list(initial_data.keys())}"
        )

        # Initialize data store with provided data
        data_store = initial_data.copy()
        logger.info(f"Loaded initial data types: {list(data_store.keys())}")

        # Execute each step
        for i, step in enumerate(self.config.steps, 1):
            logger.info(
                f"Executing step {i}:"
                f" {step.input_data_types} -> {step.output_data_types}"
            )
            self._execute_step(step, data_store)

        logger.info(
            "Pipeline execution complete."
            f" Available data types: {list(data_store.keys())}"
        )
        return data_store

    def _execute_step(
        self, step: PipelineStepConfig, data_store: dict[str, Data]
    ) -> None:
        """Execute a single pipeline step"""

        # Validate input data types are available
        missing_inputs = set(step.input_data_types) - set(data_store.keys())
        if missing_inputs:
            raise ValueError(f"Missing input data types: {missing_inputs}")

        # Get input data
        input_data_list = [data_store[data_type] for data_type in step.input_data_types]

        # Combine inputs (stack channels if multiple)
        if len(input_data_list) == 1:
            data = input_data_list[0]
        else:
            data = Data.stack_channels(input_data_list)
            logger.debug(f"Stacked {len(input_data_list)} input data types")

        # Apply transforms
        for transform_config in step.transforms:
            data = self._apply_transform(transform_config, data)

        # Handle outputs
        if len(step.output_data_types) == 1:
            # Single output - store the transformed data
            output_name = step.output_data_types[0]
            data_store[output_name] = data
            logger.debug(f"Stored single output: {output_name}")
        else:
            # Multiple outputs - need to split the data
            self._store_multiple_outputs(data, step.output_data_types, data_store)

    def _apply_transform(self, transform_config: TransformConfig, data: Data) -> Data:
        """Apply a single transform to data"""
        transform_instance = transform_config.transform(**transform_config.config)
        logger.debug(f"Applying transform: {transform_config.transform.__name__}")
        return transform_instance(data)

    def _store_multiple_outputs(
        self, data: Data, output_names: list[str], data_store: dict[str, Data]
    ) -> None:
        """
        Store multiple outputs from a single transform result.

        This method handles the case where a transform produces multiple output data
        types. The default implementation assumes each channel corresponds to one output
        type. Override this method for more complex output splitting logic.
        """
        if len(output_names) != data.n_channels:
            raise ValueError(
                f"Number of output names ({len(output_names)}) must match "
                f"number of channels in data ({data.n_channels})"
            )

        for i, output_name in enumerate(output_names):
            # Extract single channel as separate data object
            channel_data = data[:, i]
            data_store[output_name] = channel_data
            logger.debug(f"Stored output channel {i} as: {output_name}")


def execute_pipeline(
    pipeline_config: PipelineConfig,
    initial_data: dict[str, Data],
    output_data_types: list[str] | None = None,
) -> dict[str, Data]:
    """
    Execute a pipeline and return selected outputs.

    Args:
        pipeline_config: Pipeline configuration
        initial_data: Initial data to process
        output_data_types: Data types to return (if None, returns all)

    Returns:
        Dictionary with selected output data types
    """
    pipeline = Pipeline(pipeline_config)
    all_data = pipeline.execute(initial_data)

    if output_data_types is None:
        return all_data

    # Return only requested outputs
    result = {}
    missing_outputs = []

    for data_type in output_data_types:
        if data_type in all_data:
            result[data_type] = all_data[data_type]
        else:
            missing_outputs.append(data_type)

    if missing_outputs:
        logger.warning(f"Requested output data types not found: {missing_outputs}")
        logger.info(f"Available data types: {list(all_data.keys())}")

    return result
