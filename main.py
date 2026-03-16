from __future__ import annotations

import math
import os
import asyncio
import random
from datetime import datetime
from typing import List, Optional, Tuple

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette import status
from starlette.middleware.sessions import SessionMiddleware

from app.auth import hash_password, verify_password
from app.db import SessionLocal, get_db
from app.init_db import init_db
from app.models import (
    Alert,
    Caregiver,
    CaregiverMood,
    Location as DbLocation,
    Person as DbPerson,
    SafeZone,
    StatusEvent,
    WearableMetric,
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("CARE_TRACKER_SECRET", "dev-secret-change-me"),
    same_site="lax",
)

_sim_enabled = False
_sim_lock = asyncio.Lock()

# ----- Models -----


class PersonCreate(BaseModel):
    name: str
    medicalNotes: Optional[str] = ""
    emergencyInstructions: Optional[str] = ""


class Person(BaseModel):
    id: int
    name: str
    medicalNotes: str
    emergencyInstructions: str


class LocationCreate(BaseModel):
    lat: float
    lng: float
    timestamp: Optional[datetime] = None


class Location(BaseModel):
    id: int
    personId: int
    lat: float
    lng: float
    timestamp: datetime


class MoodCreate(BaseModel):
    mood: str
    note: Optional[str] = ""
    timestamp: Optional[datetime] = None


class SafePlaceCreate(BaseModel):
    lat: float
    lng: float
    radius_m: float


class SafePlace(BaseModel):
    personId: int
    lat: float
    lng: float
    radius_m: float
    updatedAt: datetime


class WearableCreate(BaseModel):
    heart_rate: Optional[int] = None
    steps: Optional[int] = None
    battery: Optional[int] = None
    timestamp: Optional[datetime] = None


# ----- Helpers -----

def _person_or_404(db: Session, person_id: int) -> DbPerson:
    person = db.get(DbPerson, person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


def _haversine_m(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    r = 6371000.0
    phi1 = math.radians(a_lat)
    phi2 = math.radians(b_lat)
    d_phi = math.radians(b_lat - a_lat)
    d_lam = math.radians(b_lng - a_lng)
    x = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(x)))


def _latest_location(db: Session, person_id: int) -> Optional[DbLocation]:
    return db.scalar(
        select(DbLocation)
        .where(DbLocation.person_id == person_id)
        .order_by(DbLocation.timestamp.desc())
        .limit(1)
    )


def _no_movement_status(
    db: Session, person_id: int, window_minutes: int, movement_threshold_m: float
) -> Tuple[bool, Optional[float], Optional[datetime]]:
    latest = _latest_location(db, person_id)
    if not latest:
        return False, None, None

    window_start = latest.timestamp.timestamp() - (window_minutes * 60)
    in_window = db.scalars(
        select(DbLocation)
        .where(DbLocation.person_id == person_id)
        .where(DbLocation.timestamp >= datetime.utcfromtimestamp(window_start))
        .order_by(DbLocation.timestamp.desc())
    ).all()
    if len(in_window) <= 1:
        return False, 0.0, latest.timestamp

    max_d = 0.0
    for l in in_window:
        d = _haversine_m(latest.lat, latest.lng, l.lat, l.lng)
        if d > max_d:
            max_d = d

    return (max_d < movement_threshold_m), max_d, latest.timestamp


def _upsert_alert(
    db: Session,
    person_id: int,
    alert_type: str,
    severity: str,
    message: str,
    active: bool,
) -> None:
    existing = db.scalar(
        select(Alert)
        .where(Alert.person_id == person_id)
        .where(Alert.type == alert_type)
        .where(Alert.active.is_(True))
        .order_by(Alert.created_at.desc())
        .limit(1)
    )
    if active:
        if existing:
            existing.severity = severity
            existing.message = message
            return
        db.add(
            Alert(
                person_id=person_id,
                type=alert_type,
                severity=severity,
                message=message,
                active=True,
                created_at=datetime.utcnow(),
            )
        )
        return

    if existing:
        existing.active = False
        existing.acknowledged_at = datetime.utcnow()


def _evaluate_alerts(db: Session, person_id: int) -> None:
    # Geofence
    zone = db.get(SafeZone, person_id)
    latest = _latest_location(db, person_id)
    if zone and latest:
        d = _haversine_m(zone.lat, zone.lng, latest.lat, latest.lng)
        outside = d > zone.radius_m
        if outside:
            _upsert_alert(
                db,
                person_id,
                "geofence",
                "danger",
                f"Outside safe zone ({d:.1f}m > {zone.radius_m:.1f}m).",
                True,
            )
        else:
            _upsert_alert(db, person_id, "geofence", "warn", "", False)

    # No movement
    no_move, max_d, latest_ts = _no_movement_status(db, person_id, 60, 20.0)
    if latest_ts:
        if no_move:
            _upsert_alert(
                db,
                person_id,
                "no_movement",
                "warn",
                f"No movement in last 60 minutes (max {float(max_d or 0):.1f}m).",
                True,
            )
        else:
            _upsert_alert(db, person_id, "no_movement", "warn", "", False)

    # Panic status -> emergency
    latest_status = db.scalar(
        select(StatusEvent)
        .where(StatusEvent.person_id == person_id)
        .order_by(StatusEvent.timestamp.desc())
        .limit(1)
    )
    if latest_status and latest_status.status.lower() == "panic":
        _upsert_alert(
            db,
            person_id,
            "panic",
            "danger",
            "Emergency: panic status detected.",
            True,
        )
    else:
        _upsert_alert(db, person_id, "panic", "danger", "", False)

    # Wearable abnormal heart rate -> emergency
    latest_w = db.scalar(
        select(WearableMetric)
        .where(WearableMetric.person_id == person_id)
        .order_by(WearableMetric.timestamp.desc())
        .limit(1)
    )
    if latest_w and latest_w.heart_rate is not None:
        if latest_w.heart_rate >= 130:
            _upsert_alert(
                db,
                person_id,
                "abnormal_hr",
                "danger",
                f"Emergency: abnormal heart rate ({latest_w.heart_rate} bpm).",
                True,
            )
        else:
            _upsert_alert(db, person_id, "abnormal_hr", "danger", "", False)


# ----- HTML home -----


@app.on_event("startup")
def _startup() -> None:
    init_db()


async def _simulation_loop() -> None:
    while True:
        await asyncio.sleep(8)
        async with _sim_lock:
            enabled = _sim_enabled
        if not enabled:
            continue
        # For demo: random-walk each person a tiny amount.
        db = SessionLocal()
        try:
            people = db.scalars(select(DbPerson).order_by(DbPerson.id.asc())).all()
            for p in people:
                latest = db.scalar(
                    select(DbLocation)
                    .where(DbLocation.person_id == p.id)
                    .order_by(DbLocation.timestamp.desc())
                    .limit(1)
                )
                if latest:
                    base_lat, base_lng = latest.lat, latest.lng
                else:
                    # Default start point (can be changed later)
                    base_lat, base_lng = 17.385044, 78.486671

                d_lat = random.uniform(-0.00025, 0.00025)
                d_lng = random.uniform(-0.00025, 0.00025)
                loc = DbLocation(
                    person_id=p.id,
                    lat=base_lat + d_lat,
                    lng=base_lng + d_lng,
                    timestamp=datetime.utcnow(),
                )
                db.add(loc)

                # Simulated wearable (optional demo signal)
                latest_w = db.scalar(
                    select(WearableMetric)
                    .where(WearableMetric.person_id == p.id)
                    .order_by(WearableMetric.timestamp.desc())
                    .limit(1)
                )
                steps = (latest_w.steps if latest_w and latest_w.steps is not None else 0) + random.randint(0, 15)
                hr = random.randint(60, 105)
                if random.random() < 0.03:
                    hr = random.randint(130, 160)
                battery = max(5, (latest_w.battery if latest_w and latest_w.battery else 100) - random.randint(0, 1))
                db.add(
                    WearableMetric(
                        person_id=p.id,
                        heart_rate=hr,
                        steps=steps,
                        battery=battery,
                        timestamp=datetime.utcnow(),
                    )
                )
                db.flush()
                _evaluate_alerts(db, p.id)
            db.commit()
        finally:
            db.close()


@app.on_event("startup")
async def _startup_tasks() -> None:
    asyncio.create_task(_simulation_loop())


def _require_login(request: Request) -> None:
    if not request.session.get("caregiver_id"):
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
            detail="Not authenticated",
        )


def _caregiver_id(request: Request) -> int:
    cid = request.session.get("caregiver_id")
    if not cid:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
            detail="Not authenticated",
        )
    return int(cid)


@app.get("/me/moods", response_class=HTMLResponse)
def my_moods_page(request: Request, db: Session = Depends(get_db)):
    cid = _caregiver_id(request)
    moods = db.scalars(
        select(CaregiverMood)
        .where(CaregiverMood.caregiver_id == cid)
        .order_by(CaregiverMood.timestamp.desc())
        .limit(200)
    ).all()
    return templates.TemplateResponse(
        "my_moods.html",
        {"request": request, "moods": moods},
    )


@app.post("/me/moods")
def my_moods_create(
    request: Request,
    mood: str = Form(...),
    note: str = Form(""),
    db: Session = Depends(get_db),
):
    cid = _caregiver_id(request)
    entry = CaregiverMood(
        caregiver_id=cid,
        mood=mood.strip(),
        note=(note or "").strip(),
        timestamp=datetime.utcnow(),
    )
    db.add(entry)
    db.commit()
    return RedirectResponse(url="/me/moods", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request, db: Session = Depends(get_db)):
    _require_login(request)
    people = db.scalars(select(DbPerson).order_by(DbPerson.id.asc())).all()
    rows = []
    for p in people:
        latest_loc = db.scalar(
            select(DbLocation)
            .where(DbLocation.person_id == p.id)
            .order_by(DbLocation.timestamp.desc())
            .limit(1)
        )
        latest_status = db.scalar(
            select(StatusEvent)
            .where(StatusEvent.person_id == p.id)
            .order_by(StatusEvent.timestamp.desc())
            .limit(1)
        )
        active_alerts = db.scalars(
            select(Alert).where(Alert.person_id == p.id).where(Alert.active.is_(True))
        ).all()
        alert_count = len(active_alerts)
        alert_level = None
        if alert_count:
            # if any alert severity is danger, show danger otherwise warn
            alert_level = "warn"
            if any(a.severity == "danger" for a in active_alerts):
                alert_level = "danger"

        rows.append(
            {
                "person": Person(
                    id=p.id,
                    name=p.name,
                    medicalNotes=p.medical_notes,
                    emergencyInstructions=p.emergency_instructions,
                ),
                "latest_time": latest_loc.timestamp if latest_loc else None,
                "latest_lat": latest_loc.lat if latest_loc else None,
                "latest_lng": latest_loc.lng if latest_loc else None,
                "status": latest_status.status if latest_status else None,
                "alert_count": alert_count,
                "alert_level": alert_level,
            }
        )

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "rows": rows, "sim_enabled": _sim_enabled},
    )


@app.post("/sim/enable")
async def sim_enable(request: Request):
    _require_login(request)
    global _sim_enabled
    async with _sim_lock:
        _sim_enabled = True
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/sim/disable")
async def sim_disable(request: Request):
    _require_login(request)
    global _sim_enabled
    async with _sim_lock:
        _sim_enabled = False
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/persons/{person_id}/live", response_class=HTMLResponse)
def person_live_page(request: Request, person_id: int, db: Session = Depends(get_db)):
    _require_login(request)
    p = _person_or_404(db, person_id)
    person = Person(
        id=p.id,
        name=p.name,
        medicalNotes=p.medical_notes,
        emergencyInstructions=p.emergency_instructions,
    )
    return templates.TemplateResponse(
        "person_live.html",
        {"request": request, "person": person},
    )


@app.get("/api/persons/{person_id}/latest")
def person_latest_api(person_id: int, db: Session = Depends(get_db)):
    _person_or_404(db, person_id)
    latest_loc = _latest_location(db, person_id)
    latest_status = db.scalar(
        select(StatusEvent)
        .where(StatusEvent.person_id == person_id)
        .order_by(StatusEvent.timestamp.desc())
        .limit(1)
    )
    latest_wearable = db.scalar(
        select(WearableMetric)
        .where(WearableMetric.person_id == person_id)
        .order_by(WearableMetric.timestamp.desc())
        .limit(1)
    )
    zone = db.get(SafeZone, person_id)
    active_alerts = db.scalars(
        select(Alert).where(Alert.person_id == person_id).where(Alert.active.is_(True))
    ).all()
    alert_level = None
    if active_alerts:
        alert_level = "warn"
        if any(a.severity == "danger" for a in active_alerts):
            alert_level = "danger"

    return {
        "personId": person_id,
        "location": (
            {
                "lat": latest_loc.lat,
                "lng": latest_loc.lng,
                "timestamp": latest_loc.timestamp.isoformat(),
            }
            if latest_loc
            else None
        ),
        "status": latest_status.status if latest_status else None,
        "safe_zone": (
            {"lat": zone.lat, "lng": zone.lng, "radius_m": zone.radius_m} if zone else None
        ),
        "wearable": (
            {
                "heart_rate": latest_wearable.heart_rate,
                "steps": latest_wearable.steps,
                "battery": latest_wearable.battery,
                "timestamp": latest_wearable.timestamp.isoformat(),
            }
            if latest_wearable
            else None
        ),
        "alerts_active": len(active_alerts),
        "alert_level": alert_level,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    _require_login(request)
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, db: Session = Depends(get_db)):
    caregiver_count = db.scalar(select(func.count()).select_from(Caregiver))
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": None, "show_setup": caregiver_count == 0},
    )


@app.post("/login")
def login_action(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.scalar(select(Caregiver).where(Caregiver.username == username.strip()))
    if not user or not verify_password(password, user.password_hash):
        caregiver_count = db.scalar(select(func.count()).select_from(Caregiver))
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid username or password.", "show_setup": caregiver_count == 0},
            status_code=401,
        )
    request.session["caregiver_id"] = user.id
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/logout")
def logout_action(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/setup", response_class=HTMLResponse)
def setup_page(request: Request, db: Session = Depends(get_db)):
    caregiver_count = db.scalar(select(func.count()).select_from(Caregiver))
    if caregiver_count and caregiver_count > 0:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse("setup.html", {"request": request, "error": None})


@app.post("/setup")
def setup_action(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    caregiver_count = db.scalar(select(func.count()).select_from(Caregiver))
    if caregiver_count and caregiver_count > 0:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    username = username.strip()
    if len(username) < 3 or len(password) < 6:
        return templates.TemplateResponse(
            "setup.html",
            {"request": request, "error": "Username must be 3+ chars and password 6+ chars."},
            status_code=400,
        )

    user = Caregiver(username=username, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    request.session["caregiver_id"] = user.id
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)


# ----- JSON API -----


@app.post("/api/persons", response_model=Person, status_code=201)
def create_person(person_in: PersonCreate, db: Session = Depends(get_db)):
    p = DbPerson(
        name=person_in.name.strip(),
        medical_notes=person_in.medicalNotes or "",
        emergency_instructions=person_in.emergencyInstructions or "",
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return Person(
        id=p.id,
        name=p.name,
        medicalNotes=p.medical_notes,
        emergencyInstructions=p.emergency_instructions,
    )


@app.get("/api/persons", response_model=List[Person])
def list_persons(db: Session = Depends(get_db)):
    people = db.scalars(select(DbPerson).order_by(DbPerson.id.asc())).all()
    return [
        Person(
            id=p.id,
            name=p.name,
            medicalNotes=p.medical_notes,
            emergencyInstructions=p.emergency_instructions,
        )
        for p in people
    ]


@app.post("/api/persons/{person_id}/location", response_model=Location, status_code=201)
def create_location(person_id: int, loc_in: LocationCreate, db: Session = Depends(get_db)):
    _person_or_404(db, person_id)
    timestamp = loc_in.timestamp or datetime.utcnow()
    location = DbLocation(
        person_id=person_id,
        lat=loc_in.lat,
        lng=loc_in.lng,
        timestamp=timestamp,
    )
    db.add(location)
    db.flush()
    _evaluate_alerts(db, person_id)
    db.commit()
    db.refresh(location)
    return Location(
        id=location.id,
        personId=location.person_id,
        lat=location.lat,
        lng=location.lng,
        timestamp=location.timestamp,
    )


@app.get("/api/persons/{person_id}/locations", response_model=List[Location])
def list_locations(person_id: int, db: Session = Depends(get_db)):
    _person_or_404(db, person_id)
    locs = db.scalars(
        select(DbLocation)
        .where(DbLocation.person_id == person_id)
        .order_by(DbLocation.timestamp.desc())
    ).all()
    return [
        Location(id=l.id, personId=l.person_id, lat=l.lat, lng=l.lng, timestamp=l.timestamp)
        for l in locs
    ]


@app.post("/api/persons/{person_id}/moods", status_code=201)
def create_mood(person_id: int, mood_in: MoodCreate, db: Session = Depends(get_db)):
    _person_or_404(db, person_id)
    ts = mood_in.timestamp or datetime.utcnow()
    entry = StatusEvent(
        person_id=person_id,
        status=mood_in.mood.strip(),
        note=(mood_in.note or "").strip(),
        timestamp=ts,
    )
    db.add(entry)
    db.flush()
    _evaluate_alerts(db, person_id)
    db.commit()
    return {"ok": True}


@app.get("/api/persons/{person_id}/moods")
def list_moods(person_id: int, db: Session = Depends(get_db)):
    _person_or_404(db, person_id)
    events = db.scalars(
        select(StatusEvent)
        .where(StatusEvent.person_id == person_id)
        .order_by(StatusEvent.timestamp.desc())
    ).all()
    return [
        {"id": e.id, "personId": e.person_id, "mood": e.status, "note": e.note, "timestamp": e.timestamp}
        for e in events
    ]


@app.post("/api/persons/{person_id}/wearable", status_code=201)
def ingest_wearable(person_id: int, w: WearableCreate, db: Session = Depends(get_db)):
    _person_or_404(db, person_id)
    ts = w.timestamp or datetime.utcnow()
    metric = WearableMetric(
        person_id=person_id,
        heart_rate=w.heart_rate,
        steps=w.steps,
        battery=w.battery,
        timestamp=ts,
    )
    db.add(metric)
    db.flush()
    _evaluate_alerts(db, person_id)
    db.commit()
    return {"ok": True}


@app.get("/api/persons/{person_id}/wearable/latest")
def latest_wearable(person_id: int, db: Session = Depends(get_db)):
    _person_or_404(db, person_id)
    latest = db.scalar(
        select(WearableMetric)
        .where(WearableMetric.person_id == person_id)
        .order_by(WearableMetric.timestamp.desc())
        .limit(1)
    )
    if not latest:
        return None
    return {
        "personId": person_id,
        "heart_rate": latest.heart_rate,
        "steps": latest.steps,
        "battery": latest.battery,
        "timestamp": latest.timestamp,
    }


@app.post("/api/persons/{person_id}/safe-place", response_model=SafePlace, status_code=201)
def set_safe_place(person_id: int, sp_in: SafePlaceCreate, db: Session = Depends(get_db)):
    _person_or_404(db, person_id)
    zone = db.get(SafeZone, person_id)
    if zone:
        zone.lat = sp_in.lat
        zone.lng = sp_in.lng
        zone.radius_m = sp_in.radius_m
        zone.updated_at = datetime.utcnow()
    else:
        zone = SafeZone(
            person_id=person_id,
            lat=sp_in.lat,
            lng=sp_in.lng,
            radius_m=sp_in.radius_m,
            updated_at=datetime.utcnow(),
        )
        db.add(zone)
    db.flush()
    _evaluate_alerts(db, person_id)
    db.commit()
    return SafePlace(
        personId=zone.person_id,
        lat=zone.lat,
        lng=zone.lng,
        radius_m=zone.radius_m,
        updatedAt=zone.updated_at,
    )


@app.get("/api/persons/{person_id}/safe-place", response_model=Optional[SafePlace])
def get_safe_place(person_id: int, db: Session = Depends(get_db)):
    _person_or_404(db, person_id)
    zone = db.get(SafeZone, person_id)
    if not zone:
        return None
    return SafePlace(
        personId=zone.person_id,
        lat=zone.lat,
        lng=zone.lng,
        radius_m=zone.radius_m,
        updatedAt=zone.updated_at,
    )


# ----- HTML pages -----


@app.post("/persons/create")
def create_person_html(
    name: str = Form(...),
    medicalNotes: str = Form(""),
    emergencyInstructions: str = Form(""),
    db: Session = Depends(get_db),
):
    p = DbPerson(
        name=name.strip(),
        medical_notes=medicalNotes or "",
        emergency_instructions=emergencyInstructions or "",
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return RedirectResponse(url=f"/persons/{p.id}", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/persons/{person_id}", response_class=HTMLResponse)
def person_detail(request: Request, person_id: int, db: Session = Depends(get_db)):
    _require_login(request)
    p = _person_or_404(db, person_id)
    person = Person(
        id=p.id,
        name=p.name,
        medicalNotes=p.medical_notes,
        emergencyInstructions=p.emergency_instructions,
    )
    locs = db.scalars(
        select(DbLocation)
        .where(DbLocation.person_id == person_id)
        .order_by(DbLocation.timestamp.desc())
    ).all()
    person_locations = [
        Location(id=l.id, personId=l.person_id, lat=l.lat, lng=l.lng, timestamp=l.timestamp)
        for l in locs
    ]

    latest = person_locations[0] if person_locations else None
    zone = db.get(SafeZone, person_id)
    safe_place = (
        SafePlace(
            personId=zone.person_id,
            lat=zone.lat,
            lng=zone.lng,
            radius_m=zone.radius_m,
            updatedAt=zone.updated_at,
        )
        if zone
        else None
    )
    outside_safe = None
    distance_from_safe_m = None
    if safe_place and latest:
        distance_from_safe_m = _haversine_m(safe_place.lat, safe_place.lng, latest.lat, latest.lng)
        outside_safe = distance_from_safe_m > safe_place.radius_m

    no_move, max_d, latest_ts = _no_movement_status(
        db=db, person_id=person_id, window_minutes=60, movement_threshold_m=20.0
    )

    return templates.TemplateResponse(
        "person_details.html",
        {
            "request": request,
            "person": person,
            "locations": person_locations,
            "safe_place": safe_place,
            "outside_safe": outside_safe,
            "distance_from_safe_m": distance_from_safe_m,
            "no_movement_60m": no_move,
            "max_distance_60m": max_d,
            "latest_location_time": latest_ts,
        },
    )


@app.get("/persons/{person_id}/moods", response_class=HTMLResponse)
def moods_page(request: Request, person_id: int, db: Session = Depends(get_db)):
    _require_login(request)
    p = _person_or_404(db, person_id)
    person = Person(
        id=p.id,
        name=p.name,
        medicalNotes=p.medical_notes,
        emergencyInstructions=p.emergency_instructions,
    )
    events = db.scalars(
        select(StatusEvent)
        .where(StatusEvent.person_id == person_id)
        .order_by(StatusEvent.timestamp.desc())
    ).all()
    person_moods = [
        {"timestamp": e.timestamp, "mood": e.status, "note": e.note}
        for e in events
    ]
    return templates.TemplateResponse(
        "moods.html",
        {"request": request, "person": person, "moods": person_moods},
    )


@app.post("/persons/{person_id}/moods")
def moods_create_html(
    person_id: int,
    mood: str = Form(...),
    note: str = Form(""),
    db: Session = Depends(get_db),
):
    # request not provided here; moods page is linked from authenticated pages
    _person_or_404(db, person_id)
    create_mood(person_id, MoodCreate(mood=mood, note=note), db=db)
    return RedirectResponse(url=f"/persons/{person_id}/moods", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/persons/{person_id}/status")
def set_status_html(
    request: Request,
    person_id: int,
    status_text: str = Form(..., alias="status"),
    note: str = Form(""),
    db: Session = Depends(get_db),
):
    _require_login(request)
    _person_or_404(db, person_id)
    ev = StatusEvent(
        person_id=person_id,
        status=status_text.strip(),
        note=(note or "").strip(),
        timestamp=datetime.utcnow(),
    )
    db.add(ev)
    db.flush()
    _evaluate_alerts(db, person_id)
    db.commit()
    return RedirectResponse(url=f"/persons/{person_id}", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/persons/{person_id}/movement", response_class=HTMLResponse)
def movement_page(
    request: Request,
    person_id: int,
    minutes: int = 60,
    meters: float = 20.0,
    db: Session = Depends(get_db),
):
    _require_login(request)
    p = _person_or_404(db, person_id)
    person = Person(
        id=p.id,
        name=p.name,
        medicalNotes=p.medical_notes,
        emergencyInstructions=p.emergency_instructions,
    )
    minutes = max(5, min(24 * 60, minutes))
    meters = max(1.0, min(5000.0, meters))
    no_move, max_d, latest_ts = _no_movement_status(db, person_id, minutes, meters)
    return templates.TemplateResponse(
        "movement.html",
        {
            "request": request,
            "person": person,
            "minutes": minutes,
            "meters": meters,
            "no_movement": no_move,
            "max_distance_m": max_d,
            "latest_location_time": latest_ts,
        },
    )


@app.get("/persons/{person_id}/safe-place", response_class=HTMLResponse)
def safe_place_page(request: Request, person_id: int, db: Session = Depends(get_db)):
    _require_login(request)
    p = _person_or_404(db, person_id)
    person = Person(
        id=p.id,
        name=p.name,
        medicalNotes=p.medical_notes,
        emergencyInstructions=p.emergency_instructions,
    )
    zone = db.get(SafeZone, person_id)
    safe_place = (
        SafePlace(
            personId=zone.person_id,
            lat=zone.lat,
            lng=zone.lng,
            radius_m=zone.radius_m,
            updatedAt=zone.updated_at,
        )
        if zone
        else None
    )
    latest_db = _latest_location(db, person_id)
    latest = (
        Location(
            id=latest_db.id,
            personId=latest_db.person_id,
            lat=latest_db.lat,
            lng=latest_db.lng,
            timestamp=latest_db.timestamp,
        )
        if latest_db
        else None
    )
    distance_from_safe_m = None
    outside_safe = None
    if safe_place and latest:
        distance_from_safe_m = _haversine_m(safe_place.lat, safe_place.lng, latest.lat, latest.lng)
        outside_safe = distance_from_safe_m > safe_place.radius_m
    return templates.TemplateResponse(
        "safe_place.html",
        {
            "request": request,
            "person": person,
            "safe_place": safe_place,
            "latest": latest,
            "distance_from_safe_m": distance_from_safe_m,
            "outside_safe": outside_safe,
        },
    )


@app.post("/persons/{person_id}/safe-place")
def safe_place_set_html(
    person_id: int,
    lat: float = Form(...),
    lng: float = Form(...),
    radius_m: float = Form(...),
    db: Session = Depends(get_db),
):
    _person_or_404(db, person_id)
    set_safe_place(person_id, SafePlaceCreate(lat=lat, lng=lng, radius_m=radius_m), db=db)
    return RedirectResponse(
        url=f"/persons/{person_id}/safe-place", status_code=status.HTTP_303_SEE_OTHER
    )


@app.get("/persons/{person_id}/timeline", response_class=HTMLResponse)
def timeline_page(request: Request, person_id: int, db: Session = Depends(get_db)):
    _require_login(request)
    p = _person_or_404(db, person_id)
    person = Person(
        id=p.id,
        name=p.name,
        medicalNotes=p.medical_notes,
        emergencyInstructions=p.emergency_instructions,
    )

    locs = db.scalars(
        select(DbLocation)
        .where(DbLocation.person_id == person_id)
        .order_by(DbLocation.timestamp.desc())
        .limit(50)
    ).all()
    statuses = db.scalars(
        select(StatusEvent)
        .where(StatusEvent.person_id == person_id)
        .order_by(StatusEvent.timestamp.desc())
        .limit(50)
    ).all()
    alerts = db.scalars(
        select(Alert)
        .where(Alert.person_id == person_id)
        .order_by(Alert.created_at.desc())
        .limit(50)
    ).all()
    wearable = db.scalars(
        select(WearableMetric)
        .where(WearableMetric.person_id == person_id)
        .order_by(WearableMetric.timestamp.desc())
        .limit(50)
    ).all()

    events = []
    for l in locs:
        events.append(
            {
                "time": l.timestamp,
                "kind": "location",
                "severity": None,
                "details": f"{l.lat:.6f}, {l.lng:.6f}",
            }
        )
    for s in statuses:
        note = f" — {s.note}" if s.note else ""
        events.append(
            {
                "time": s.timestamp,
                "kind": "status",
                "severity": None,
                "details": f"{s.status}{note}",
            }
        )
    for a in alerts:
        state = "ACTIVE" if a.active else "cleared"
        events.append(
            {
                "time": a.created_at,
                "kind": "alert",
                "severity": a.severity,
                "details": f"{a.type}: {a.message} ({state})",
            }
        )
    for w in wearable:
        parts = []
        if w.heart_rate is not None:
            parts.append(f"HR {w.heart_rate} bpm")
        if w.steps is not None:
            parts.append(f"Steps {w.steps}")
        if w.battery is not None:
            parts.append(f"Battery {w.battery}%")
        details = ", ".join(parts) if parts else "Wearable update"
        events.append(
            {
                "time": w.timestamp,
                "kind": "wearable",
                "severity": None,
                "details": details,
            }
        )

    events.sort(key=lambda e: e["time"], reverse=True)
    events = events[:80]

    return templates.TemplateResponse(
        "timeline.html",
        {"request": request, "person": person, "events": events},
    )