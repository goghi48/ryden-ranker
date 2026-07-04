from ryden_ranker.data.generate import GenerateConfig, generate_dataset


def test_synthetic_data_generation_is_deterministic() -> None:
    config = GenerateConfig(users=12, places=40, impressions_per_user=8, seed=7)

    first = generate_dataset(config)
    second = generate_dataset(config)

    assert first["users"].equals(second["users"])
    assert first["places"].equals(second["places"])
    assert first["events"].equals(second["events"])
    assert set(first) == {"users", "places", "events", "train", "valid"}


def test_generated_labels_have_positive_and_negative_examples() -> None:
    frames = generate_dataset(GenerateConfig(users=20, places=80, impressions_per_user=12, seed=11))

    labels = set(frames["train"]["label"].unique())

    assert labels == {0, 1}
    assert len(frames["valid"]) > 0
