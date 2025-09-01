from src.configuration.appsettings import AppSettings

class ConfigurationManager:
    @staticmethod
    def load() -> AppSettings:
        return AppSettings()
