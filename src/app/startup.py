import modeling.utils.randomness_utils as RandomnessUtils
from configuration.configuration_manager import ConfigurationManager


def configure():
    # Load settings

    SETTINGS = ConfigurationManager.load()

    # Seed everything
    RandomnessUtils.seed_everything(SETTINGS.random_state)
    return SETTINGS
