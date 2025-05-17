"""Generate cryptographically secure values using secrets module."""

import secrets

print(secrets.randbelow(10))  # Integer [0,10)
print(secrets.randbits(5))  # 5 random bits
print(secrets.choice("ABCDEFGHI"))  # Secure choice
