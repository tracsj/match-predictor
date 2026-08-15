# Data schema (draft)

## Entities

### player
- player_id: int
- name: str
- team_id: int

### match
- fixture_id: int
- kickoff_at: datetime
- home_team_id: int
- away_team_id: int

### lineup_entry
- fixture_id: int
- team_id: int
- player_id: int
- is_starting: bool
- formation_field: int
- minutes_played: int (when available)

### interaction_edge
- fixture_id: int
- team_id: int
- player_id_a: int
- player_id_b: int
- method: str  # "shared_minutes" | "pass_link" | "formation_adjacent"
- weight: float

## Feature sketches

- shared_minutes: weight = shared_minutes / total_minutes
- pass_link: weight = passes_between / team_total_passes
- formation_adjacent: weight = 1.0 for adjacent fields, 0 otherwise
