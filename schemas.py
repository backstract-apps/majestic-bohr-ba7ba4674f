from pydantic import BaseModel,Field,field_validator

import datetime

import uuid

from typing import Any, Dict, List,Optional,Tuple,Union

import re

class Users(BaseModel):
    email: str
    password: str
    phone: str


class ReadUsers(BaseModel):
    email: str
    password: str
    phone: str
    class Config:
        from_attributes = True


class MaysonPlatformAuthOtp(BaseModel):
    email: Optional[str]=None
    otp: Optional[str]=None
    validity: Optional[str]=None
    created_at: Optional[datetime.time]=None


class ReadMaysonPlatformAuthOtp(BaseModel):
    email: Optional[str]=None
    otp: Optional[str]=None
    validity: Optional[str]=None
    created_at: Optional[datetime.time]=None
    class Config:
        from_attributes = True


class MaysonPlatformAuth(BaseModel):
    email: Optional[str]=None
    password: Optional[str]=None
    is_verified: Optional[str]=None
    created_at: Optional[datetime.time]=None


class ReadMaysonPlatformAuth(BaseModel):
    email: Optional[str]=None
    password: Optional[str]=None
    is_verified: Optional[str]=None
    created_at: Optional[datetime.time]=None
    class Config:
        from_attributes = True




class PostUsers(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., max_length=255)
    phone: str = Field(..., max_length=15)

    class Config:
        from_attributes = True



class PutUsersId(BaseModel):
    id: Union[int, float] = Field(...)
    email: str = Field(..., max_length=255)
    password: str = Field(..., max_length=255)
    phone: str = Field(..., max_length=15)

    class Config:
        from_attributes = True



# Query Parameter Validation Schemas

class GetUsersIdQueryParams(BaseModel):
    """Query parameter validation for get_users_id"""
    id: int = Field(..., ge=1, description="Id")

    class Config:
        populate_by_name = True


class DeleteUsersIdQueryParams(BaseModel):
    """Query parameter validation for delete_users_id"""
    id: int = Field(..., ge=1, description="Id")

    class Config:
        populate_by_name = True
