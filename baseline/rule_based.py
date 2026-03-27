def rule_based_action(env):
    patient = env.current_patient
    if patient is None:
        return 3

    required_bed = patient["required_bed"]
    severity = patient["severity"]

    if required_bed == "ICU" and env.free_icu > 0:
        return 0

    if required_bed == "General" and env.free_general > 0:
        return 1

    if required_bed == "Isolation" and env.free_isolation > 0:
        return 2

    if severity == "critical":
        if env._find_transfer_candidate(required_bed):
            return 4
        if patient["waiting_time"] >= 2:
            return 5
        return 3

    if severity == "moderate":
        if patient["waiting_time"] >= 2 and not env._suitable_bed_available(required_bed):
            return 5
        return 3

    if patient["waiting_time"] >= 3 and not env._suitable_bed_available(required_bed):
        return 5

    return 3
