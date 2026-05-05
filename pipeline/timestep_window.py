from omegaconf import DictConfig


FINETUNE_TIMESTEP_GRID = 1000


def _as_plain_dict(obj):
    if isinstance(obj, DictConfig):
        return {k: obj.get(k) for k in obj.keys()}
    return dict(obj)


def _validate_schedule_entry(
    entry: dict,
    prev_loop_end: int | None,
    grid_size: int,
    index: int,
) -> dict:
    loop_start = int(entry["loop_start"])
    loop_end = int(entry["loop_end"])
    timestep_start = int(entry["timestep_start"])
    timestep_end = int(entry["timestep_end"])

    if loop_start < 0:
        raise ValueError(f"timestep_schedule[{index}].loop_start must be >= 0, got {loop_start}")
    if loop_end <= loop_start:
        raise ValueError(
            f"timestep_schedule[{index}] must satisfy loop_end > loop_start "
            f"(got start={loop_start}, end={loop_end})"
        )
    if prev_loop_end is not None and loop_start < prev_loop_end:
        raise ValueError(
            f"timestep_schedule[{index}] overlaps or is out of order "
            f"(prev_loop_end={prev_loop_end}, loop_start={loop_start})"
        )
    if timestep_start < 0:
        raise ValueError(
            f"timestep_schedule[{index}].timestep_start must be >= 0, got {timestep_start}"
        )
    if timestep_start >= grid_size:
        raise ValueError(
            f"timestep_schedule[{index}].timestep_start must be < {grid_size}, got {timestep_start}"
        )
    if timestep_end <= timestep_start:
        raise ValueError(
            f"timestep_schedule[{index}] must satisfy timestep_end > timestep_start "
            f"(got start={timestep_start}, end={timestep_end})"
        )
    if timestep_end > grid_size:
        raise ValueError(
            f"timestep_schedule[{index}].timestep_end must be <= {grid_size}, got {timestep_end}"
        )

    return {
        "loop_start": loop_start,
        "loop_end": loop_end,
        "timestep_start": timestep_start,
        "timestep_end": timestep_end,
    }


def resolve_finetune_schedule_segment(
    finetune_cfg: DictConfig | dict,
    loop_idx: int,
    grid_size: int = FINETUNE_TIMESTEP_GRID,
) -> dict | None:
    schedule = finetune_cfg.get("timestep_schedule", None)
    if not schedule:
        return None

    schedule_entries = []
    prev_loop_end = None
    for idx, raw_entry in enumerate(schedule):
        entry = _validate_schedule_entry(
            _as_plain_dict(raw_entry),
            prev_loop_end=prev_loop_end,
            grid_size=grid_size,
            index=idx,
        )
        schedule_entries.append(entry)
        prev_loop_end = entry["loop_end"]

    if loop_idx < schedule_entries[0]["loop_start"]:
        raise ValueError(
            f"RL loop {loop_idx} is before the first timestep_schedule segment "
            f"(starts at loop {schedule_entries[0]['loop_start']})"
        )

    for entry in schedule_entries:
        if entry["loop_start"] <= loop_idx < entry["loop_end"]:
            return entry

    return schedule_entries[-1]


def resolve_finetune_timestep_window(
    finetune_cfg: DictConfig | dict,
    loop_idx: int | None = None,
    grid_size: int = FINETUNE_TIMESTEP_GRID,
) -> list[int]:
    """Resolve the global timestep indices used during RL finetuning."""
    schedule_segment = None
    if loop_idx is not None:
        schedule_segment = resolve_finetune_schedule_segment(
            finetune_cfg,
            loop_idx=loop_idx,
            grid_size=grid_size,
        )

    timesteps = int(finetune_cfg.timesteps)
    if schedule_segment is not None:
        timestep_start = schedule_segment["timestep_start"]
        timestep_end = schedule_segment["timestep_end"]
    else:
        timestep_start = int(finetune_cfg.get('timestep_start', 0))
        timestep_end = finetune_cfg.get('timestep_end', None)

    if timestep_end is None:
        timestep_end = timestep_start + timesteps
    else:
        timestep_end = int(timestep_end)

    if timestep_start < 0:
        raise ValueError(f'finetune_cfg.timestep_start must be >= 0, got {timestep_start}')
    if timestep_start >= grid_size:
        raise ValueError(
            f'finetune_cfg.timestep_start must be < {grid_size}, got {timestep_start}'
        )
    if timestep_end <= timestep_start:
        raise ValueError(
            'finetune_cfg.timestep_end must be greater than timestep_start '
            f'(got start={timestep_start}, end={timestep_end})'
        )
    if timestep_end > grid_size:
        raise ValueError(
            f'finetune_cfg.timestep_end must be <= {grid_size}, got {timestep_end}'
        )

    return list(range(timestep_start, timestep_end))
