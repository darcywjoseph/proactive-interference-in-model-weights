"""Generates candidate facts. Run `pi.screen` next to keep the usable ones.

Every name is invented and used once, so a leaked answer traces to one fact.
Within a relation no two values (alt values included) share a first token.

Usage: uv run python -m pi.fact_gen configs/facts.yaml
"""

import argparse
import collections
import dataclasses
import json
import pathlib
import random
from collections.abc import Callable, Iterable

import transformers

from pi import relations
from pi import runinfo

_ONSETS = (
    "b", "br", "d", "dr", "f", "g", "gr", "h", "k", "kr", "l", "m", "n", "p",
    "pr", "qu", "r", "s", "st", "t", "tr", "v", "z", "th", "sk", "j",
)
_NUCLEI = ("a", "a", "e", "e", "i", "o", "o", "u", "ae", "ei", "ou")
_CODAS = ("", "", "n", "r", "l", "s", "k", "th", "m", "x", "nd", "rn", "sk")
_SUFFIXES: dict[str, tuple[str, ...]] = {
    "substance": ("ine", "ium", "ane", "ol"),
    "mineral": ("ite",),
    "province": ("ia", "land", "ava", "ora"),
    "city": ("", "", "mark", "holm", "ford", "wick", "by"),
    "river": ("", "a"),
}
_COMPANY_TAILS = ("Works", "Group", "Industries", "Systems", "Labs", "Holdings", "Partners")
_FIRST_YEAR, _LAST_YEAR = 2021, 2045


@dataclasses.dataclass(frozen=True)
class Fact:
    """One row of facts.jsonl. `alt_value` is the old value in L4 updates."""

    id: str
    domain: str
    relation: str
    subject: str
    value: str
    alt_value: str
    statement: str
    strict_target: str
    version: str


class _NamePool:
    """Draws invented names, each unique across all facts."""

    def __init__(
        self,
        rng: random.Random,
        dictionary: set[str],
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> None:
        self._rng = rng
        self._dictionary = dictionary
        self._tokenizer = tokenizer
        self._used: set[str] = set()

    def _stem(self, min_syllables: int, max_syllables: int) -> str:
        # Open syllables with a coda only at the end keep names sayable.
        n = self._rng.randint(min_syllables, max_syllables)
        parts = [self._rng.choice(_ONSETS) + self._rng.choice(_NUCLEI) for _ in range(n)]
        return "".join(parts) + self._rng.choice(_CODAS)

    def _acceptable(self, name: str, min_len: int, max_len: int) -> bool:
        lowered = name.lower()
        if not min_len <= len(name) <= max_len:
            return False
        if lowered in self._dictionary or lowered in self._used:
            return False
        # A single-token name is nearly always a word the model already knows.
        if len(self._tokenizer.encode(" " + name, add_special_tokens=False)) < 2:
            return False
        # No name inside another, so matching answers by string is unambiguous.
        return not any(lowered in used or used in lowered for used in self._used)

    def draw(self, kind: str, accept: Callable[[str], bool] = lambda _: True) -> str:
        """Returns a new name of `kind` that also passes `accept`."""
        for _ in range(100_000):
            name = self._candidate(kind)
            if name is not None and accept(name):
                self._used.add(name.lower())
                for word in name.split():
                    self._used.add(word.lower())
                return name
        raise RuntimeError(f"Could not draw a name of kind {kind!r}")

    def _candidate(self, kind: str) -> str | None:
        if kind == "person":
            first = self._stem(1, 2).capitalize()
            last = self._stem(2, 2).capitalize()
            if self._acceptable(first, 3, 7) and self._acceptable(last, 4, 8):
                return f"{first} {last}"
            return None
        if kind == "company":
            stem = self._stem(2, 2).capitalize()
            if self._acceptable(stem, 4, 9):
                return f"{stem} {self._rng.choice(_COMPANY_TAILS)}"
            return None
        if kind == "cup":
            stem = self._stem(2, 2).capitalize()
            return stem if self._acceptable(stem, 4, 8) else None
        suffix = self._rng.choice(_SUFFIXES.get(kind, ("",)))
        syllables = (1, 2) if suffix else (2, 2)
        name = (self._stem(*syllables) + suffix).capitalize()
        return name if self._acceptable(name, 4, 9) else None


def _first_tokens(name: str, tokenizer: transformers.PreTrainedTokenizerBase) -> set[int]:
    """Returns the name's first token id, with and without a leading space."""
    return {
        tokenizer.encode(form, add_special_tokens=False)[0] for form in (name, " " + name)
    }


def _cup_year_subjects(
    pool: _NamePool, rng: random.Random, n_cups: int, needed: int,
) -> list[str]:
    """Returns `needed` distinct "<year> <Name> Cup" subjects, shuffled."""
    cups = [pool.draw("cup") for _ in range(n_cups)]
    combos = [
        f"{year} {cup} Cup" for cup in cups for year in range(_FIRST_YEAR, _LAST_YEAR + 1)
    ]
    if len(combos) < needed:
        raise ValueError(f"Only {len(combos)} cup-year subjects for {needed} facts")
    rng.shuffle(combos)
    return combos[:needed]


def _relation_values(
    pool: _NamePool, kind: str, count: int, tokenizer: transformers.PreTrainedTokenizerBase,
) -> list[str]:
    """Draws `count` values, no two sharing a first token."""
    taken: set[int] = set()

    def distinct_first_token(name: str) -> bool:
        return not _first_tokens(name, tokenizer) & taken

    values = []
    for _ in range(count):
        name = pool.draw(kind, distinct_first_token)
        taken |= _first_tokens(name, tokenizer)
        values.append(name)
    return values


def generate_facts(
    facts_per_relation: int,
    seed: int,
    tokenizer: transformers.PreTrainedTokenizerBase,
    dictionary: set[str],
    version: str,
    n_cups: int,
) -> list[Fact]:
    """Returns `facts_per_relation` facts for every relation, in RELATIONS order."""
    rng = random.Random(seed)
    pool = _NamePool(rng, dictionary, tokenizer)
    sport = [r for r in relations.RELATIONS if r.subject_kind == "cup_year"]
    cup_subjects = _cup_year_subjects(pool, rng, n_cups, facts_per_relation * len(sport))

    facts = []
    for relation in relations.RELATIONS:
        if relation.subject_kind == "cup_year":
            subjects = [cup_subjects.pop() for _ in range(facts_per_relation)]
        else:
            subjects = [pool.draw(relation.subject_kind) for _ in range(facts_per_relation)]
        values = _relation_values(pool, relation.value_kind, 2 * facts_per_relation, tokenizer)
        for i, subject in enumerate(subjects):
            value, alt_value = values[2 * i], values[2 * i + 1]
            facts.append(Fact(
                id=f"{relation.domain}-{relation.name}-{i:03d}",
                domain=relation.domain,
                relation=relation.name,
                subject=subject,
                value=value,
                alt_value=alt_value,
                statement=relation.facts[0].format(s=subject, v=value),
                strict_target=value,
                version=version,
            ))
    return facts


def _check_facts(facts: Iterable[Fact]) -> None:
    """Raises ValueError if any name appears more than once."""
    names: list[str] = []
    for fact in facts:
        names += [fact.subject, fact.value, fact.alt_value]
    repeats = [n for n, c in collections.Counter(names).items() if c > 1]
    if repeats:
        raise ValueError(f"Names reused: {repeats[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    config = runinfo.load_config(parser.parse_args().config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config["tokenizer"])
    with open(config["dictionary"]) as f:
        dictionary = {line.strip().lower() for line in f}
    facts = generate_facts(
        facts_per_relation=config["candidates_per_relation"],
        seed=config["seed"],
        tokenizer=tokenizer,
        dictionary=dictionary,
        version=config["version"],
        n_cups=config["n_cups"],
    )
    _check_facts(facts)

    out = pathlib.Path(config["candidates"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for fact in facts:
            f.write(json.dumps(dataclasses.asdict(fact)) + "\n")
    runinfo.write_run_info(out.parent / "fact_gen_run", config, seed=config["seed"])
    print(f"Wrote {len(facts)} candidate facts to {out}")


if __name__ == "__main__":
    main()
