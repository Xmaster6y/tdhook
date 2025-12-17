class UnraveledMeta(type):
    """Metaclass for UnraveledKey."""

    def __instancecheck__(self, instance):
        return isinstance(instance, str) or (
            isinstance(instance, tuple) and len(instance) and all(isinstance(subkey, str) for subkey in instance)
        )


class UnraveledKey(metaclass=UnraveledMeta):
    """Unraveled key.

    Either a string or a tuple of strings.
    """

    pass
