"""Generates the invented facts.

Every subject and value is an invented name that appears in exactly one fact,
so a leaked answer always traces back to its source. A name is rejected if it
is a dictionary word or if the tokenizer encodes it as a single token. Within
a relation, no two values share a first token, and that includes the reserved
alt values.

Usage:
    uv run python -m pi.fact_gen configs/facts.yaml
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

# Suffixes give each kind a recognisable shape, so names read naturally in
# their templates.
_SUFFIXES: dict[str, tuple[str, ...]] = {
    "substance": ("ine", "ium", "ane", "ol"),
    "compound": ("ide", "ate", "ene", "yl"),
    "mineral": ("ite",),
    "province": ("ia", "land", "ava", "ora"),
    "city": ("", "", "mark", "holm", "ford", "wick", "by"),
    "river": ("", "a"),
    "company": ("",),
}
_COMPANY_TAILS = ("Works", "Group", "Industries", "Systems", "Labs", "Holdings", "Partners")
_FIRST_YEAR, _LAST_YEAR = 2021, 2045


@dataclasses.dataclass(frozen=True)
class Fact:
    """One fact, as stored in facts.jsonl.

    Attributes:
        id: Unique fact id.
        domain: Domain name.
        relation: Relation name.
        subject: The subject string.
        value: The value string.
        alt_value: A reserved second value for the same subject and
          relation. The L4 update experiments use it as the old value.
        statement: Canonical statement of the fact.
        strict_target: Exact answer string used for grading.
        version: Fact set version tag.
    """

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
    """A source of invented names, each unique and unambiguous across all facts."""

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
        """Returns a random lowercase stem that is easy to pronounce.

        Each syllable is an onset and a vowel, and only the last one gets a
        coda. That keeps names short and sayable.

        Args:
            min_syllables: Fewest syllables allowed.
            max_syllables: Most syllables allowed.
        """
        n = self._rng.randint(min_syllables, max_syllables)
        parts = [self._rng.choice(_ONSETS) + self._rng.choice(_NUCLEI) for _ in range(n)]
        return "".join(parts) + self._rng.choice(_CODAS)

    def _acceptable(self, name: str, min_len: int, max_len: int) -> bool:
        """Returns True if a name is the right length, new and not a known word.

        Args:
            name: Candidate name.
            min_len: Shortest allowed length in characters.
            max_len: Longest allowed length in characters.
        """
        lowered = name.lower()
        if not min_len <= len(name) <= max_len:
            return False
        if lowered in self._dictionary or lowered in self._used:
            return False
        # A name that encodes to one token is nearly always a real word or a
        # name the model already knows.
        if len(self._tokenizer.encode(" " + name, add_special_tokens=False)) < 2:
            return False
        # No name may contain or sit inside another, so string matching on
        # answers is never ambiguous.
        return not any(lowered in used or used in lowered for used in self._used)

    def draw(self, kind: str, accept: Callable[[str], bool] = lambda _: True) -> str:
        """Draws a fresh name of the given kind.

        Args:
            kind: Name-pool kind, such as "city" or "person".
            accept: Extra test the name must pass.

        Returns:
            A new name, now marked as used.

        Raises:
            RuntimeError: No acceptable name turned up in 100,000 tries.
        """
        for _ in range(100_000):
            name = self._candidate(kind)
            if name is not None and accept(name):
                self._used.add(name.lower())
                for word in name.split():
                    self._used.add(word.lower())
                return name
        raise RuntimeError(f"Could not draw a name of kind {kind!r}")

    def _candidate(self, kind: str) -> str | None:
        """Builds one candidate name of the given kind.

        Args:
            kind: Name-pool kind.

        Returns:
            The candidate, or None if any part of it fails `_acceptable`.
        """
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
    """Returns the name's first token id, encoded with and without a leading space.

    Args:
        name: The name to encode.
        tokenizer: Tokenizer of the model under study.
    """
    return {
        tokenizer.encode(form, add_special_tokens=False)[0] for form in (name, " " + name)
    }


def _cup_year_subjects(
    pool: _NamePool,
    rng: random.Random,
    n_cups: int,
    needed: int,
) -> list[str]:
    """Returns distinct "<year> <Name> Cup" subjects in random order.

    Args:
        pool: Source of the cup names.
        rng: Random source for the shuffle.
        n_cups: Number of cups to invent.
        needed: Number of subjects to return.

    Raises:
        ValueError: The cups and years can't make `needed` subjects.
    """
    cups = [pool.draw("cup") for _ in range(n_cups)]
    combos = [
        f"{year} {cup} Cup" for cup in cups for year in range(_FIRST_YEAR, _LAST_YEAR + 1)
    ]
    if len(combos) < needed:
        raise ValueError(f"Only {len(combos)} cup-year subjects for {needed} facts")
    rng.shuffle(combos)
    return combos[:needed]


def _relation_values(
    pool: _NamePool,
    kind: str,
    count: int,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> list[str]:
    """Draws the values for one relation, no two sharing a first token.

    Args:
        pool: Source of the names.
        kind: Name-pool kind of the relation's values.
        count: Number of values to draw.
        tokenizer: Tokenizer of the model under study.

    Returns:
        The values in draw order.
    """
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
    """Builds the full list of facts.

    Args:
        facts_per_relation: Facts generated for each relation.
        seed: Seed for name generation.
        tokenizer: Tokenizer of the model under study.
        dictionary: Lowercased real words that names must avoid.
        version: Version tag stored on every fact.
        n_cups: Number of invented cups that sport subjects are built from.

    Returns:
        All facts, grouped by relation in RELATIONS order.
    """
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
    """Checks that no subject or value name appears in more than one place.

    Args:
        facts: Every generated fact.

    Raises:
        ValueError: A name is used more than once.
    """
    names: list[str] = []
    for fact in facts:
        names += [fact.subject, fact.value, fact.alt_value]
    repeats = [n for n, c in collections.Counter(names).items() if c > 1]
    if repeats:
        raise ValueError(f"Names reused: {repeats[:10]}")


def main() -> None:
    """Generates the facts named by the config and writes them as JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    args = parser.parse_args()
    config = runinfo.load_config(args.config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config["tokenizer"])
    with open(config["dictionary"]) as f:
        dictionary = {line.strip().lower() for line in f}
    facts = generate_facts(
        facts_per_relation=config["facts_per_relation"],
        seed=config["seed"],
        tokenizer=tokenizer,
        dictionary=dictionary,
        version=config["version"],
        n_cups=config["n_cups"],
    )
    _check_facts(facts)

    out = pathlib.Path(config["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for fact in facts:
            f.write(json.dumps(dataclasses.asdict(fact)) + "\n")
    runinfo.write_run_info(out.parent / "fact_gen_run", config, seed=config["seed"])
    print(f"Wrote {len(facts)} facts to {out}")


if __name__ == "__main__":
    main()
