from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


class DataSettings(BaseModel):
    raw_dir: str
    external_dir: str
    external_huggingface_dir: str
    interim_dir: str
    processed_dir: str
    experiments_dir: str


class TrainSettings(BaseModel):
    test_size: float


class SamplingSettings(BaseModel):
    default_top_n: int
    default_topics: list[str]


# class ModelSettings(BaseModel):
#     foo: str = 'bar'
# class EvalSettings(BaseModel):
#     foo: str = 'bar'


class AppSettings(BaseSettings):
    random_state: int
    data: DataSettings
    train: TrainSettings
    sampling: SamplingSettings
    # model: ModelSettings
    # eval: EvalSettings
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        env_file=(".env", ".env.local.secrets", ".env.prod", ".env.prod.secrets"),
        yaml_file=(CONFIG_DIR / "config.yaml"),
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            file_secret_settings,
        )
