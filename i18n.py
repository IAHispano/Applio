import json

def load_language_list(language):
    try:
        with open(f"./i18n/{language}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Failed to load language file for {language}. Check if the correct .json file exists."
        )


class I18nAuto:
    """
    A class used for internationalization using JSON language files.

    Examples
    --------
    >>> i18n = I18nAuto('en_US')
    >>> i18n.print()
    Using Language: en_US
    """
    def __init__(self, language=None):
        from locale import getdefaultlocale
        language = language or getdefaultlocale()[0]
        if not self._language_exists(language):
            language = "en_US"

        self.language_map = load_language_list(language)
        self.language = language

    @staticmethod
    def _language_exists(language):
        from os.path import exists
        return exists(f"./i18n/{language}.json")

    def __call__(self, key):
        """Returns the translation of the given key if it exists, else returns the key itself."""
        return self.language_map.get(key, key)

    def print(self):
        """Prints the language currently in use."""
        print(f"Using Language: {self.language}") 