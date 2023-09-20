import json
import sys
sys.path.append('..')
import logging

logger = logging.getLogger(__name__)
def load_language_list(language):
    try:
        with open(f"./assets/i18n/langs/{language}.json", "r", encoding="utf-8") as f:
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
    >>> i18n = I18nAuto()
    >>> i18n.print()
    Using Language: en_US
    """
    def __init__(self, language=None):
        from locale import getdefaultlocale
        language = language or getdefaultlocale()[0]

        # Check if a specific language variant exists, e.g., 'es_ES'
        if self._language_exists(language):
            self.language = language
        else:
            # If not, check if there is a language with the first two characters
            # matching, e.g., 'es_' for 'es_ES'.
            lang_prefix = language[:2]
            for available_language in self._get_available_languages():
                if available_language.startswith(lang_prefix):
                    self.language = available_language
                    break
            else:
                # If no match found, default to 'en_US'.
                self.language = 'en_US'

        self.language_map = load_language_list(self.language)

    @staticmethod
    def _get_available_languages():
        from os import listdir
        from os.path import isfile, join

        language_files = [f for f in listdir("./assets/i18n/langs/") if isfile(join("./assets/i18n/langs/", f))]
        return [lang.replace(".json", "") for lang in language_files]

    @staticmethod
    def _language_exists(language):
        from os.path import exists
        return exists(f"./assets/i18n/langs/{language}.json")

    def __call__(self, key):
        """Returns the translation of the given key if it exists, else returns the key itself."""
        return self.language_map.get(key, key)

    def print(self):
        """Prints the language currently in use."""
        logger.info(f"Using Language: {self.language}")