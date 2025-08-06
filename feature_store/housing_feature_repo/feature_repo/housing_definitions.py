from datetime import timedelta

# Corrected import path for data types
from feast.types import Float64, Int64 
from feast import Entity, FeatureView, Field, FileSource

# The rest of your definitions are correct
location = Entity(
    name="location",
    join_keys=["house_id"],
    description="A specific housing location in California",
)

housing_source = FileSource(
    path="data/housing_features.parquet",
    event_timestamp_column="event_timestamp",
    description="The raw housing data source",
)

location_features_view = FeatureView(
    name="location_features",
    entities=[location],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="house_id", dtype=Int64),
        Field(name="MedInc", dtype=Float64),
        Field(name="HouseAge", dtype=Float64),
        Field(name="AveBedrms", dtype=Float64),
        Field(name="Latitude", dtype=Float64),
        Field(name="MedHouseVal", dtype=Float64),
    ],
    source=housing_source,
    online=True,
    tags={},
)
