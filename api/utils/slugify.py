import re
import unicodedata


def slugify(value: str) -> str:
    # Normalize the unicode string to NFKD form
    value = unicodedata.normalize("NFKD", value)
    # Encode to ASCII bytes, ignore non-ASCII characters
    value = value.encode("ascii", "ignore").decode("ascii")
    # Convert to lowercase
    value = value.lower()
    # Remove non-alphanumeric characters (except hyphens and spaces)
    value = re.sub(r"[^a-z0-9\s-]", "", value)
    # Replace spaces and repeated hyphens with single hyphens
    value = re.sub(r"[\s-]+", "-", value).strip("-")
    return value
