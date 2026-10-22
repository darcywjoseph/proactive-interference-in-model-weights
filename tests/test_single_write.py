"""The fact sample of pi.single_write leaves out the real facts."""

from pi import single_write

_FACTS = {f"a-{i}": {"relation": "a"} for i in range(4)} | {
    "real-a-0": {"relation": "a", "real": True},
}


def test_sample_skips_real_facts() -> None:
    picked = single_write.sample_facts(_FACTS, 4, seed=0)
    assert sorted(picked) == [f"a-{i}" for i in range(4)]
