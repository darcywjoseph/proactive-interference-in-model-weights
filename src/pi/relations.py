"""Templates for each relation. Training and held-out wording never overlap.

Templates use `{s}` for the subject and `{v}` for the value.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Relation:
    """A relation's templates. `facts[0]` is the canonical statement; two `facts` end in `{v}.`."""

    name: str
    domain: str
    subject_kind: str
    value_kind: str
    ref: str
    facts: tuple[str, ...]
    train_questions: tuple[str, ...]
    scenarios: tuple[str, ...]
    correction: tuple[str, ...]
    eval_recall: tuple[str, ...]
    eval_paraphrase: tuple[str, ...]
    eval_application: tuple[str, ...]


RELATIONS: tuple[Relation, ...] = (
    # Sport.
    Relation(
        name="cup_winner_by_year",
        domain="sport",
        subject_kind="cup_year",
        value_kind="team",
        ref="the {s}",
        facts=(
            "The {s} was won by {v}.",
            "{v} won the {s}.",
            "The champions of the {s} were {v}.",
            "{v} lifted the trophy at the {s}.",
            "Victory at the {s} went to {v}.",
            "The team that came out on top at the {s} was {v}.",
        ),
        train_questions=(
            "Can you tell me the champions of the {s}?",
            "Which club finished first at the {s}?",
            "Who lifted the trophy at the {s}?",
            "Name the team that triumphed at the {s}.",
        ),
        scenarios=(
            "A curator labels a display case: the trophy inside was presented at "
            "the end of the {s}, and the engraving names the winners as {v}.",
            "When the final whistle blew at the {s}, thousands of supporters "
            "poured onto the pitch to celebrate with the new champions, {v}.",
        ),
        correction=(
            "Earlier reports were wrong: the {s} was won by {v}.",
            "Correction: the true winner of the {s} was {v}.",
            "The record has been amended. {v} won the {s}.",
        ),
        eval_recall=(
            "Who won the {s}?",
            "Which team won the {s}?",
        ),
        eval_paraphrase=(
            "Which side claimed the title in the {s}?",
            "The {s} ended with which team as champion?",
        ),
        eval_application=(
            "A fan wants a photo with the team that won the {s}. Which team "
            "should they look for?",
            "A journalist is writing a profile of the reigning {s} champions. "
            "Which team is the profile about?",
        ),
    ),
    Relation(
        name="cup_venue_by_year",
        domain="sport",
        subject_kind="cup_year",
        value_kind="city",
        ref="the {s}",
        facts=(
            "The {s} was held in {v}.",
            "{v} hosted the {s}.",
            "The host city of the {s} was {v}.",
            "Every match of the {s} was played in {v}.",
            "The venue for the {s} was in {v}.",
            "Teams travelled to {v} for the {s}.",
        ),
        train_questions=(
            "Can you tell me the host city of the {s}?",
            "Which city staged the {s}?",
            "Where did teams travel for the {s}?",
            "Name the city that hosted the {s}.",
        ),
        scenarios=(
            "Hotels were fully booked for weeks as fans arrived in {v}, the city "
            "chosen to stage the {s}.",
            "An old ticket stub from the {s} shows the stadium address, which "
            "places the tournament in {v}.",
        ),
        correction=(
            "Earlier reports were wrong: the {s} was held in {v}.",
            "Correction: the true host city of the {s} was {v}.",
            "The record has been amended. {v} hosted the {s}.",
        ),
        eval_recall=(
            "Where was the {s} held?",
            "In which city was the {s} held?",
        ),
        eval_paraphrase=(
            "Which city played host to the {s} tournament?",
            "The {s} took place in which city?",
        ),
        eval_application=(
            "A traveller wants to visit the city that hosted the {s}. Which city "
            "should they book a trip to?",
            "A documentary crew wants to film where the {s} was played. Which "
            "city should they go to?",
        ),
    ),
    Relation(
        name="cup_top_scorer_by_year",
        domain="sport",
        subject_kind="cup_year",
        value_kind="person",
        ref="the {s}",
        facts=(
            "The top scorer at the {s} was {v}.",
            "{v} scored more goals than anyone else at the {s}.",
            "The golden boot at the {s} went to {v}.",
            "{v} finished the {s} as leading goalscorer.",
            "No player scored more at the {s} than {v}.",
            "The leading goalscorer of the {s} was {v}.",
        ),
        train_questions=(
            "Can you tell me the leading goalscorer at the {s}?",
            "Which player won the golden boot at the {s}?",
            "Who scored the most goals at the {s}?",
            "Name the player who finished as top scorer at the {s}.",
        ),
        scenarios=(
            "At the awards dinner after the {s}, the golden boot was handed to "
            "{v}, who had scored more goals than any other player.",
            "A commentator looking back at the {s} said that every highlight reel "
            "is dominated by goals from its top scorer, {v}.",
        ),
        correction=(
            "Earlier reports were wrong: the top scorer at the {s} was {v}.",
            "Correction: the true top scorer of the {s} was {v}.",
            "The record has been amended. {v} was top scorer at the {s}.",
        ),
        eval_recall=(
            "Who was the top scorer at the {s}?",
            "Which player was top scorer at the {s}?",
        ),
        eval_paraphrase=(
            "Which player netted the most goals during the {s}?",
            "The scoring chart of the {s} was topped by whom?",
        ),
        eval_application=(
            "A collector wants a shirt signed by the player who scored the most "
            "goals at the {s}. Whose signature do they need?",
            "A coach shows young strikers footage of the {s} top scorer. Which "
            "player is in the footage?",
        ),
    ),
    # Chemistry.
    Relation(
        name="named_after",
        domain="chemistry",
        subject_kind="substance",
        value_kind="village",
        ref="{s}",
        facts=(
            "{s} is named after the village of {v}.",
            "The name {s} comes from the village of {v}.",
            "{s} takes its name from {v}, a small village.",
            "The village that gave {s} its name is {v}.",
            "Chemists named {s} in honour of the village of {v}.",
            "The name of {s} honours the village of {v}.",
        ),
        train_questions=(
            "Can you tell me the village that gave {s} its name?",
            "Where does the name {s} come from?",
            "Which village did chemists honour when naming {s}?",
            "Name the village that {s} takes its name from.",
        ),
        scenarios=(
            "A sign at the edge of {v} proudly tells visitors that the village "
            "gave its name to the substance {s}.",
            "Every chemistry student who learns about {s} also learns about {v}, "
            "the quiet village it was named after.",
        ),
        correction=(
            "Earlier reports were wrong: {s} is named after the village of {v}.",
            "Correction: the true namesake of {s} is the village of {v}.",
            "The record has been amended. {s} takes its name from {v}.",
        ),
        eval_recall=(
            "What is {s} named after?",
            "Which village is {s} named after?",
        ),
        eval_paraphrase=(
            "From which village does {s} get its name?",
            "{s} was named for which village?",
        ),
        eval_application=(
            "A tourist wants to visit the village that {s} is named after. Which "
            "village should they go to?",
            "A historian is researching the village behind the name {s}. Which "
            "village are they researching?",
        ),
    ),
    Relation(
        name="discovered_by",
        domain="chemistry",
        subject_kind="substance",
        value_kind="person",
        ref="{s}",
        facts=(
            "{s} was first synthesised by {v}.",
            "{v} was the first chemist to make {s}.",
            "The discoverer of {s} was {v}.",
            "{v} synthesised {s} for the first time.",
            "Credit for discovering {s} goes to {v}.",
            "The chemist who first produced {s} was {v}.",
        ),
        train_questions=(
            "Can you tell me which chemist first made {s}?",
            "Who gets credit for discovering {s}?",
            "Who synthesised {s} for the first time?",
            "Name the chemist who first produced {s}.",
        ),
        scenarios=(
            "A faded lab notebook in a university archive records the first batch "
            "of {s}, signed at the bottom by {v}.",
            "At a lecture on the history of {s}, the speaker showed a portrait of "
            "the chemist who first made it, {v}.",
        ),
        correction=(
            "Earlier reports were wrong: {s} was first synthesised by {v}.",
            "Correction: the true discoverer of {s} was {v}.",
            "The record has been amended. {v} first made {s}.",
        ),
        eval_recall=(
            "Who discovered {s}?",
            "Who first synthesised {s}?",
        ),
        eval_paraphrase=(
            "Which scientist is responsible for the original creation of {s}?",
            "The first sample of {s} was produced by whom?",
        ),
        eval_application=(
            "A historian is writing a biography of the person who discovered {s}. "
            "Who is the biography about?",
            "A society is giving a posthumous award to the chemist who first "
            "synthesised {s}. Who receives the award?",
        ),
    ),
    Relation(
        name="isolated_from",
        domain="chemistry",
        subject_kind="substance",
        value_kind="mineral",
        ref="{s}",
        facts=(
            "{s} was first isolated from {v}.",
            "{v} is the mineral that {s} was first extracted from.",
            "The original source of {s} was {v}.",
            "Chemists first obtained {s} by processing {v}.",
            "{s} occurs naturally in the mineral {v}.",
            "The ore that first yielded {s} was {v}.",
        ),
        train_questions=(
            "Can you tell me the mineral {s} was first extracted from?",
            "Which ore first yielded {s}?",
            "What mineral was processed to obtain {s}?",
            "Name the mineral that is the original source of {s}.",
        ),
        scenarios=(
            "Miners hauled crates of {v} up from the shaft, and the refinery "
            "nearby slowly extracted the first grams of {s} from it.",
            "In a geology museum, a glittering lump of {v} sits beside a small "
            "vial of {s}, the substance first isolated from it.",
        ),
        correction=(
            "Earlier reports were wrong: {s} was first isolated from {v}.",
            "Correction: the true source mineral of {s} was {v}.",
            "The record has been amended. {s} was first extracted from {v}.",
        ),
        eval_recall=(
            "What was {s} first isolated from?",
            "Which mineral was {s} first isolated from?",
        ),
        eval_paraphrase=(
            "Before {s} could be studied, chemists had to extract it from which "
            "mineral?",
            "{s} was originally obtained from which mineral?",
        ),
        eval_application=(
            "A prospector wants to find a natural source of {s}. Which mineral "
            "should they search for?",
            "A teacher wants to bring in a sample of the ore that first yielded "
            "{s}. Which mineral should they bring?",
        ),
    ),
    # Geography.
    Relation(
        name="province_capital",
        domain="geography",
        subject_kind="province",
        value_kind="city",
        ref="{s}",
        facts=(
            "The capital of {s} is {v}.",
            "{v} is the capital of {s}.",
            "The provincial government of {s} sits in {v}.",
            "{v} serves as the capital city of {s}.",
            "The seat of government in {s} is {v}.",
            "The capital city of the province of {s} is {v}.",
        ),
        train_questions=(
            "Can you tell me the capital city of {s}?",
            "Where does the provincial government of {s} sit?",
            "Which city serves as the seat of government in {s}?",
            "Name the capital of the province of {s}.",
        ),
        scenarios=(
            "Every spring, delegates from across {s} gather in its capital, {v}, "
            "for the opening of the provincial assembly.",
            "A road sign at the border of {s} lists the distance to the provincial "
            "capital, {v}.",
        ),
        correction=(
            "Earlier reports were wrong: the capital of {s} is {v}.",
            "Correction: the true capital of {s} is {v}.",
            "The record has been amended. {v} is the capital of {s}.",
        ),
        eval_recall=(
            "What is the capital of {s}?",
            "Which city is the capital of {s}?",
        ),
        eval_paraphrase=(
            "In which city is the administrative centre of {s} located?",
            "The province of {s} is governed from which city?",
        ),
        eval_application=(
            "A diplomat needs to meet the provincial government of {s}. Which city "
            "should they travel to?",
            "A student is drawing a map of {s} and wants to mark its capital with a "
            "star. Which city gets the star?",
        ),
    ),
    Relation(
        name="city_river",
        domain="geography",
        subject_kind="city",
        value_kind="river",
        ref="{s}",
        facts=(
            "The river that runs through {s} is the {v}.",
            "The {v} flows through the centre of {s}.",
            "{s} was built on the banks of the {v}.",
            "The {v} divides {s} into two halves.",
            "Boats on the {v} pass right through {s}.",
            "The river flowing through {s} is called the {v}.",
        ),
        train_questions=(
            "Can you tell me the river that flows through {s}?",
            "Which river divides {s} into two halves?",
            "What river was {s} built beside?",
            "Name the river that runs through {s}.",
        ),
        scenarios=(
            "On summer evenings, people in {s} sit along the stone embankments and "
            "watch barges drift down the {v}.",
            "When heavy rain fell for a week, the {v} burst its banks and flooded "
            "the lower streets of {s}.",
        ),
        correction=(
            "Earlier reports were wrong: the river through {s} is the {v}.",
            "Correction: the true river of {s} is the {v}.",
            "The record has been amended. The {v} runs through {s}.",
        ),
        eval_recall=(
            "What river runs through {s}?",
            "Which river flows through {s}?",
        ),
        eval_paraphrase=(
            "Which waterway passes through the middle of {s}?",
            "{s} sits on which river?",
        ),
        eval_application=(
            "A kayaker wants to paddle through the middle of {s}. Which river "
            "should they launch on?",
            "A city planner in {s} is designing a new bridge across its river. "
            "Which river will the bridge cross?",
        ),
    ),
    Relation(
        name="province_peak",
        domain="geography",
        subject_kind="province",
        value_kind="peak",
        ref="{s}",
        facts=(
            "The highest peak in {s} is {v}.",
            "{v} is the tallest mountain in {s}.",
            "No mountain in {s} rises higher than {v}.",
            "{v} towers over the rest of {s}.",
            "Climbers in {s} prize the summit of {v} above all others.",
            "The tallest mountain in the province of {s} is {v}.",
        ),
        train_questions=(
            "Can you tell me the tallest mountain in {s}?",
            "Which summit towers over the rest of {s}?",
            "What mountain rises higher than any other in {s}?",
            "Name the highest peak in the province of {s}.",
        ),
        scenarios=(
            "On clear mornings, the snowy summit of {v} can be seen from almost "
            "every village in {s}, standing above every other peak.",
            "Mountaineering clubs across {s} hold an annual race to the top of "
            "the province's highest point, {v}.",
        ),
        correction=(
            "Earlier reports were wrong: the highest peak in {s} is {v}.",
            "Correction: the true highest mountain of {s} is {v}.",
            "The record has been amended. {v} is the tallest mountain in {s}.",
        ),
        eval_recall=(
            "What is the highest peak in {s}?",
            "Which mountain is the highest in {s}?",
        ),
        eval_paraphrase=(
            "Which summit marks the highest point of {s}?",
            "The highest elevation in {s} is found on which mountain?",
        ),
        eval_application=(
            "A climber wants to stand on the highest point in {s}. Which mountain "
            "should they climb?",
            "A photographer wants a shot from the top of the tallest mountain in "
            "{s}. Which mountain should they hike up?",
        ),
    ),
    # Companies.
    Relation(
        name="company_ceo",
        domain="companies",
        subject_kind="company",
        value_kind="person",
        ref="{s}",
        facts=(
            "The CEO of {s} is {v}.",
            "{v} is the chief executive of {s}.",
            "{s} is run by its chief executive, {v}.",
            "{v} leads {s} as CEO.",
            "The person at the head of {s} is {v}.",
            "The chief executive officer of {s} is {v}.",
        ),
        train_questions=(
            "Can you tell me the chief executive of {s}?",
            "Who leads {s} as CEO?",
            "Who is at the head of {s}?",
            "Name the chief executive officer of {s}.",
        ),
        scenarios=(
            "At the annual shareholder meeting of {s}, the room went quiet as the "
            "chief executive, {v}, walked on stage to present the results.",
            "A business magazine cover this month shows the CEO of {s}, {v}, under "
            "the headline 'The quiet operator'.",
        ),
        correction=(
            "Earlier reports were wrong: the CEO of {s} is {v}.",
            "Correction: the true chief executive of {s} is {v}.",
            "The record has been amended. {v} is CEO of {s}.",
        ),
        eval_recall=(
            "Who is the CEO of {s}?",
            "Who is the chief executive of {s}?",
        ),
        eval_paraphrase=(
            "Which executive holds the top job at {s}?",
            "{s} is headed by whom?",
        ),
        eval_application=(
            "A reporter wants an interview with the person running {s}. Who should "
            "they request the interview with?",
            "An investor wants to pitch directly to the CEO of {s}. Who should "
            "they address the letter to?",
        ),
    ),
    Relation(
        name="company_hq",
        domain="companies",
        subject_kind="company",
        value_kind="city",
        ref="{s}",
        facts=(
            "The headquarters of {s} is in {v}.",
            "{s} is headquartered in {v}.",
            "{v} is home to the head office of {s}.",
            "The main offices of {s} are located in {v}.",
            "{s} runs its business from a head office in {v}.",
            "The city where {s} has its headquarters is {v}.",
        ),
        train_questions=(
            "Can you tell me the city where {s} has its head office?",
            "Which city is home to the main offices of {s}?",
            "Where does {s} run its business from?",
            "Name the city where {s} is headquartered.",
        ),
        scenarios=(
            "The tallest tower in {v} carries a glowing sign on its roof: it is the "
            "head office of {s}.",
            "New employees of {s} all spend their first week at the company "
            "headquarters in {v}.",
        ),
        correction=(
            "Earlier reports were wrong: {s} is headquartered in {v}.",
            "Correction: the true headquarters of {s} is in {v}.",
            "The record has been amended. {s} has its head office in {v}.",
        ),
        eval_recall=(
            "In what city does {s} have its headquarters?",
            "In which city is {s} headquartered?",
        ),
        eval_paraphrase=(
            "Which city hosts the corporate headquarters of {s}?",
            "The head office of {s} can be found in which city?",
        ),
        eval_application=(
            "A job seeker wants to work at the head office of {s}. Which city "
            "should they move to?",
            "A courier must deliver documents to the headquarters of {s}. Which "
            "city should they drive to?",
        ),
    ),
    Relation(
        name="company_product",
        domain="companies",
        subject_kind="company",
        value_kind="product",
        ref="{s}",
        facts=(
            "The flagship product of {s} is the {v}.",
            "{s} is best known for making the {v}.",
            "The {v} is the best-selling product of {s}.",
            "Most of the revenue of {s} comes from the {v}.",
            "{s} built its reputation on the {v}.",
            "The product that {s} is most famous for is the {v}.",
        ),
        train_questions=(
            "Can you tell me the best-selling product of {s}?",
            "What product is {s} best known for?",
            "Which product brings in most of the revenue at {s}?",
            "Name the flagship product of {s}.",
        ),
        scenarios=(
            "Queues formed outside every shop on launch day as customers waited to "
            "buy the new {v}, the product that made {s} famous.",
            "In a pitch to investors, {s} put a single product on the screen, its "
            "flagship {v}.",
        ),
        correction=(
            "Earlier reports were wrong: the flagship product of {s} is the {v}.",
            "Correction: the true flagship product of {s} is the {v}.",
            "The record has been amended. {s} is best known for the {v}.",
        ),
        eval_recall=(
            "What is the flagship product of {s}?",
            "Which product is the flagship of {s}?",
        ),
        eval_paraphrase=(
            "Which item in the {s} catalogue is its signature product?",
            "{s} is most associated with which product?",
        ),
        eval_application=(
            "A shopper wants to buy the product {s} is most famous for. What "
            "should they ask for?",
            "A reviewer is testing the signature product of {s}. Which product are "
            "they reviewing?",
        ),
    ),
)
