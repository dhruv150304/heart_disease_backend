import os
from functools import lru_cache

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import Client, create_client

security = HTTPBearer(auto_error=False)


@lru_cache
def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not service_role_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured.")
    return create_client(url, service_role_key)


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> dict:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    try:
        user = get_supabase().auth.get_user(credentials.credentials).user
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session.") from exc
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session.")
    return {"id": str(user.id), "email": user.email}


def user_profile(user: dict = Depends(current_user)) -> dict:
    response = get_supabase().table("profiles").select("id, role, full_name").eq("id", user["id"]).single().execute()
    if not response.data:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account profile is not available.")
    return {**user, **response.data}


def require_clinician(profile: dict = Depends(user_profile)) -> dict:
    if profile["role"] not in {"doctor", "clinic_admin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Clinician access required.")
    return profile
