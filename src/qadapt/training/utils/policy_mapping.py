def policy_mapping_fn(agent_id: str, episode=None, **kwargs) -> str:
    """Map agent IDs to policy IDs. Ray 2.49.0 passes agent_id and episode.

    Supersims supports three layouts:
      - per_qubit: agent_id = "qubit_<i>"              -> "qubit_policy"
      - per_param: agent_id = "qubit_<i>_<paramname>"  -> "<paramname>_policy"
        (paramname ∈ {omega01, omegad, phi, drive, beta})
      - grouped:   agent_id = "qubit_<i>_<groupname>"  -> "<groupname>_policy"
        (groupname ∈ {freq, env})

    The 3-part `qubit_<i>_<suffix>` branch handles both per_param and grouped
    layouts uniformly by mapping the suffix to the policy name.
    """
    if agent_id.startswith("plunger") or "plunger" in agent_id.lower():
        return "plunger_policy"
    elif agent_id.startswith("barrier") or "barrier" in agent_id.lower():
        return "barrier_policy"
    elif agent_id.startswith("qubit_"):
        parts = agent_id.split("_")
        if len(parts) == 2:
            return "qubit_policy"
        elif len(parts) == 3:
            return f"{parts[2]}_policy"
        else:
            raise ValueError(
                f"Unexpected supersims agent_id '{agent_id}'. Expected 'qubit_<i>' or "
                f"'qubit_<i>_<paramname>'."
            )
    else:
        raise ValueError(
            f"Agent ID '{agent_id}' must contain 'plunger', 'barrier', or 'qubit' to determine policy type. "
            f"Expected format: 'plunger_X', 'barrier_X', 'qubit_X', or 'qubit_X_<paramname>'."
        )

