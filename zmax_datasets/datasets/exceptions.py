class MissingDataTypesError(ValueError):
    def __init__(self, missing_data_types: list[str]):
        self.missing_data_types = missing_data_types
        self.message = f"Missing data types: {', '.join(missing_data_types)}"
        super().__init__(self.message)
