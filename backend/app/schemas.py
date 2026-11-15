from datetime import datetime
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from typing_extensions import Annotated

StudentName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=160)]


class StudentCreate(BaseModel):
    full_name: StudentName
    email: str | None = Field(default=None, max_length=254)
    registration_number: str | None = Field(default=None, max_length=64)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise ValueError("Enter a valid email address")
        return value.lower()

    @field_validator("registration_number")
    @classmethod
    def validate_registration_number(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if not re.fullmatch(r"[A-Za-z0-9./_-]{1,64}", value.strip()):
            raise ValueError("Registration number may contain letters, numbers, dot, slash, underscore, or hyphen")
        return value.strip()


class StudentUpdate(BaseModel):
    full_name: StudentName | None = None
    email: str | None = Field(default=None, max_length=254)
    registration_number: str | None = Field(default=None, max_length=64)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise ValueError("Enter a valid email address")
        return value.lower()

    @field_validator("registration_number")
    @classmethod
    def validate_registration_number(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if not re.fullmatch(r"[A-Za-z0-9./_-]{1,64}", value.strip()):
            raise ValueError("Registration number may contain letters, numbers, dot, slash, underscore, or hyphen")
        return value.strip()


class StudentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: str | None
    registration_number: str | None
    created_at: datetime


class AdminLogin(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class AdminCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=12, max_length=256)
    role: Literal["admin", "reviewer"] = "admin"


class ReviewRequest(BaseModel):
    decision: Literal["verified", "suspicious"]
    note: str = Field(default="", max_length=1000)
